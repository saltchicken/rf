import argparse
import configparser
import asyncio
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
import struct
import zmq
import zmq.asyncio
from pathlib import Path

config_dir = f"{Path(__file__).parent}/config"


class Receiver:
    def __init__(self, args):
        self.port = args.port
        self.sample_rate = args.sample_rate
        self.center_freq = args.center_freq
        self.buffer_size = args.buffer_size
        self.gain = args.gain

        self.sdr = None
        self.rxStream = None
        self.stream_task = None
        self.setup_sdr()

        self.stop_event = asyncio.Event()

        self.ctx = zmq.asyncio.Context()

        # PUB socket for streaming samples
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://0.0.0.0:{self.port}")
        print(f"ZeroMQ PUB server broadcasting on port {self.port}")

        # REP socket for config control
        self.rep_port = self.port + 1
        self.rep_socket = self.ctx.socket(zmq.REP)
        self.rep_socket.bind(f"tcp://0.0.0.0:{self.rep_port}")
        print(f"ZeroMQ REP server listening on port {self.rep_port}")

    def get_current_settings(self):
        return {
            "sample_rate": self.sample_rate,
            "center_freq": self.center_freq,
            "gain": self.gain,
        }

    def setup_sdr(self, driver=None):
        if driver:
            args = SoapySDR.SoapySDRKwargs()
            args["driver"] = driver
        else:
            results = SoapySDR.Device.enumerate()
            if not results:
                raise RuntimeError("No SDR devices found.")
            args = results[0]

        self.sdr = SoapySDR.Device(args)

        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
        self.sdr.setGain(SOAPY_SDR_RX, 0, self.gain)

    def close(self):
        print("Receiver cleanup started.")
        try:
            if self.sdr:
                try:
                    if self.rxStream is not None:
                        self.sdr.deactivateStream(self.rxStream)
                        self.sdr.closeStream(self.rxStream)
                        self.rxStream = None
                except Exception as e:
                    print(f"Error closing SDR stream: {e}")
        finally:
            self.sdr = None
            self.pub_socket.close(linger=0)
            self.rep_socket.close(linger=0)
            self.ctx.term()
            print("Receiver cleanup finished.")

    async def start_stream(self):
        if self.rxStream is not None:
            print("Stream already running.")
            return
        self.rxStream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rxStream)
        self.stream_task = asyncio.create_task(self._stream_loop())

    async def stop_stream(self):
        if self.rxStream is None:
            return
        self.sdr.deactivateStream(self.rxStream)
        self.sdr.closeStream(self.rxStream)
        self.rxStream = None
        if self.stream_task:
            self.stream_task.cancel()
            try:
                await self.stream_task
            except asyncio.CancelledError:
                pass

    async def _stream_loop(self):
        loop = asyncio.get_running_loop()
        try:
            while not self.stop_event.is_set():
                buff = np.empty(self.buffer_size, np.complex64)
                sr = await loop.run_in_executor(
                    None, self.sdr.readStream, self.rxStream, [buff], self.buffer_size
                )
                if sr.ret > 0:
                    sample_bytes = buff[: sr.ret].tobytes()
                    if len(sample_bytes) != self.buffer_size * 8:
                        print(f"Short read: {len(sample_bytes)} bytes. Skipping")
                        continue
                    topic = b"samples"
                    length = struct.pack("!I", len(sample_bytes))
                    await self.pub_socket.send_multipart([topic, length, sample_bytes])
                else:
                    await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            print("Stream loop cancelled.")
        except Exception as e:
            print(f"Error in stream loop: {e}")

    async def reconfigure(self, setting_name, value):
        await self.stop_stream()
        if setting_name == "center_freq":
            self.center_freq = float(value)
            self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
        elif setting_name == "sample_rate":
            self.sample_rate = float(value)
            self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
        elif setting_name == "gain":
            self.gain = float(value)
            self.sdr.setGain(SOAPY_SDR_RX, 0, self.gain)
        await self.start_stream()

    async def control_listener(self):
        while not self.stop_event.is_set():
            try:
                message = await self.rep_socket.recv_json()

                if "settings" in message:
                    await self.rep_socket.send_json(self.get_current_settings())
                    continue

                if "center_freq" in message:
                    await self.reconfigure("center_freq", message["center_freq"])

                if "sample_rate" in message:
                    await self.reconfigure("sample_rate", message["sample_rate"])

                if "gain" in message:
                    await self.reconfigure("gain", message["gain"])

                response = {"status": "ok"}
                await self.rep_socket.send_json(response)

            except asyncio.CancelledError:
                print("Control Listener: Server shutdown requested (Ctrl+C)")
                self.stop_event.set()
            except Exception as e:
                print(f"Control listener error: {e}")
                await self.rep_socket.send_json({"status": "error", "message": str(e)})

        print("Control listener stopped.")


async def run():
    config = configparser.ConfigParser()
    config.read(f"{config_dir}/config.ini")

    parser = argparse.ArgumentParser(description="FM receiver and demodulator")
    parser.add_argument("--host", type=str, default=config["Network"]["HOST"])
    parser.add_argument("--port", type=int, default=config.getint("Network", "PORT"))
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=config.getfloat("Processing", "SAMPLE_RATE"),
    )
    parser.add_argument(
        "--freq_offset",
        type=float,
        default=config.getfloat("Demodulation", "FREQ_OFFSET"),
    )
    parser.add_argument(
        "--chunk_size", type=int, default=config.getint("Processing", "CHUNK_SIZE")
    )
    parser.add_argument(
        "--center_freq", type=float, default=config.getfloat("Server", "CENTER_FREQ")
    )
    parser.add_argument(
        "--buffer_size", type=int, default=config.getint("Server", "BUFFER_SIZE")
    )
    parser.add_argument("--gain", type=float, default=config.getfloat("Server", "GAIN"))

    args = parser.parse_args()

    receiver = Receiver(args)
    try:
        await asyncio.gather(receiver.start_stream(), receiver.control_listener())
    except asyncio.CancelledError:
        print("Server shutdown requested (Ctrl+C)")
        receiver.stop_event.set()
    finally:
        await receiver.stop_stream()
        receiver.close()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
