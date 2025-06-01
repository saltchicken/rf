import argparse
import asyncio
import configparser
import struct
from pathlib import Path

import numpy as np
import zmq
import zmq.asyncio
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants


config_dir = f"{Path(__file__).parent}/config"


class Transceiver:
    def __init__(self, args):
        self.port = args.port
        self.sample_rate = args.sample_rate
        self.center_freq = args.center_freq
        self.buffer_size = args.buffer_size
        self.gain = args.gain

        self.sdr = None
        self.rxStream = None
        self.txStream = None

        self.stop_event = asyncio.Event()

        self.ctx = zmq.asyncio.Context()

        # PUB socket for streaming RX samples
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://0.0.0.0:{self.port}")
        print(f"ZeroMQ PUB server broadcasting on port {self.port}")

        # REP socket for config/control
        self.rep_port = self.port + 1
        self.rep_socket = self.ctx.socket(zmq.REP)
        self.rep_socket.bind(f"tcp://0.0.0.0:{self.rep_port}")
        print(f"ZeroMQ REP server listening on port {self.rep_port}")

        # PULL socket for transmitting TX samples
        self.tx_port = self.port + 2
        self.tx_socket = self.ctx.socket(zmq.PULL)
        self.tx_socket.bind(f"tcp://0.0.0.0:{self.tx_port}")
        print(f"ZeroMQ PULL server for TX bound on port {self.tx_port}")

        self.setup_sdr()

    def setup_sdr(self):
        results = SoapySDR.Device.enumerate()
        if not results:
            raise RuntimeError("No SDR devices found.")
        args = results[0]
        self.sdr = SoapySDR.Device(args)

        # RX configuration
        self.sdr.setAntenna(SOAPY_SDR_RX, 0, "LNAW")  # LimeSDR specific
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
        self.sdr.setGain(SOAPY_SDR_RX, 0, self.gain)

        # TX configuration
        self.sdr.setAntenna(SOAPY_SDR_TX, 0, "BAND2")  # Adjust for your SDR
        self.sdr.setSampleRate(SOAPY_SDR_TX, 0, self.sample_rate)
        self.sdr.setFrequency(SOAPY_SDR_TX, 0, self.center_freq)
        self.sdr.setGain(SOAPY_SDR_TX, 0, self.gain)

    def get_current_settings(self):
        return {
            "sample_rate": self.sample_rate,
            "center_freq": self.center_freq,
            "gain": self.gain,
        }

    async def stream_samples(self):
        self.rxStream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rxStream)
        loop = asyncio.get_running_loop()

        try:
            while not self.stop_event.is_set():
                buff = np.empty(self.buffer_size, np.complex64)
                sr = await loop.run_in_executor(
                    None,
                    self.sdr.readStream,
                    self.rxStream,
                    [buff],
                    self.buffer_size,
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
            print("Stream Samples: Shutdown requested")
            self.stop_event.set()
        except Exception as e:
            print(f"Error during stream processing: {e}")
        finally:
            await asyncio.sleep(1)
            self.close()

    async def control_listener(self):
        while not self.stop_event.is_set():
            try:
                message = await self.rep_socket.recv_json()

                if "settings" in message:
                    await self.rep_socket.send_json(self.get_current_settings())
                    continue

                if "gain" in message:
                    self.gain = float(message["gain"])
                    self.sdr.setGain(SOAPY_SDR_RX, 0, self.gain)
                    self.sdr.setGain(SOAPY_SDR_TX, 0, self.gain)

                if "center_freq" in message:
                    self.center_freq = float(message["center_freq"])
                    print(f"Setting center frequency to {self.center_freq}")
                    self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
                    self.sdr.setFrequency(SOAPY_SDR_TX, 0, self.center_freq)

                if "sample_rate" in message:
                    self.sample_rate = float(message["sample_rate"])
                    self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
                    self.sdr.setSampleRate(SOAPY_SDR_TX, 0, self.sample_rate)

                response = {"status": "ok", "settings": self.get_current_settings()}
                await self.rep_socket.send_json(response)

            except asyncio.CancelledError:
                print("Control Listener: Shutdown requested")
                self.stop_event.set()

            except Exception as e:
                print(f"Control listener error: {e}")
                await self.rep_socket.send_json({"status": "error", "message": str(e)})

        print("Control listener stopped.")

    async def transmit_samples(self):
        self.txStream = self.sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
        self.sdr.activateStream(self.txStream)
        print("TX stream activated.")

        try:
            while not self.stop_event.is_set():
                msg = await self.tx_socket.recv()
                samples = np.frombuffer(msg, dtype=np.complex64)
                sr = self.sdr.writeStream(self.txStream, [samples], len(samples))
                if sr.ret < 0:
                    print(f"TX write error: {sr.ret}")
        except asyncio.CancelledError:
            print("Transmit Samples: Shutdown requested")
            self.stop_event.set()
        except Exception as e:
            print(f"TX error: {e}")

    def close(self):
        print("Cleanup started.")
        try:
            if self.sdr:
                if self.rxStream:
                    try:
                        self.sdr.deactivateStream(self.rxStream)
                        self.sdr.closeStream(self.rxStream)
                    except Exception as e:
                        print(f"Error closing RX stream: {e}")

                if self.txStream:
                    try:
                        self.sdr.deactivateStream(self.txStream)
                        self.sdr.closeStream(self.txStream)
                    except Exception as e:
                        print(f"Error closing TX stream: {e}")
        finally:
            self.sdr = None
            self.pub_socket.close(linger=0)
            self.rep_socket.close(linger=0)
            self.tx_socket.close(linger=0)
            self.ctx.term()
            print("Cleanup finished.")


async def run():
    config = configparser.ConfigParser()
    config.read(f"{config_dir}/config.ini")

    parser = argparse.ArgumentParser(description="SDR Transceiver with ZMQ")
    parser.add_argument("--host", type=str, default=config["Network"]["HOST"])
    parser.add_argument("--port", type=int, default=int(config["Network"]["PORT"]))
    parser.add_argument(
        "--sample_rate", type=float, default=float(config["Processing"]["SAMPLE_RATE"])
    )
    parser.add_argument(
        "--freq_offset",
        type=float,
        default=float(config["Demodulation"]["FREQ_OFFSET"]),
    )
    parser.add_argument(
        "--chunk_size", type=int, default=int(config["Processing"]["CHUNK_SIZE"])
    )
    parser.add_argument(
        "--center_freq", type=float, default=float(config["Server"]["CENTER_FREQ"])
    )
    parser.add_argument(
        "--buffer_size", type=int, default=int(config["Server"]["BUFFER_SIZE"])
    )
    parser.add_argument("--gain", type=float, default=float(config["Server"]["GAIN"]))
    args = parser.parse_args()

    device = Transceiver(args)
    try:
        await asyncio.gather(
            device.stream_samples(),
            device.control_listener(),
            device.transmit_samples(),
        )
    except asyncio.CancelledError:
        print("Shutdown requested (Ctrl+C)")
    finally:
        device.close()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
