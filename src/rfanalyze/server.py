import argparse
import configparser
import asyncio
import uhd
import numpy as np
import struct
import zmq
import zmq.asyncio

from pathlib import Path
from datetime import timedelta

config_dir = f"{Path(__file__).parent}/config"


class Receiver:
    def __init__(self, args):
        self.port = args.port
        self.sample_rate = args.sample_rate
        self.center_freq = args.center_freq
        self.buffer_size = args.buffer_size
        self.gain = args.gain

        self.usrp = None
        self.streamer = None
        self.metadata = uhd.types.RXMetadata()
        self.setup_usrp()

        self.stop_event = asyncio.Event()

        self.ctx = zmq.asyncio.Context()

        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://0.0.0.0:{self.port}")
        print(f"ZeroMQ PUB server broadcasting on port {self.port}")

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

    def setup_usrp(self):
        self.usrp = uhd.usrp.MultiUSRP("serial=31BADBF")

        self.usrp.set_rx_rate(self.sample_rate)
        self.usrp.set_rx_freq(self.center_freq)
        self.usrp.set_rx_gain(self.gain)

        stream_args = uhd.usrp.StreamArgs("fc32")  # Complex float32
        self.streamer = self.usrp.get_rx_stream(stream_args)

        self.recv_buffer = np.zeros((self.buffer_size,), dtype=np.complex64)

        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = True
        self.usrp.issue_stream_cmd(stream_cmd)

    def close(self):
        print("Receiver cleanup started.")
        try:
            if self.usrp:
                stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
                self.usrp.issue_stream_cmd(stop_cmd)
        except Exception as e:
            print(f"Error stopping stream: {e}")
        finally:
            self.pub_socket.close(linger=0)
            self.rep_socket.close(linger=0)
            self.ctx.term()
            print("Receiver cleanup finished.")

    async def stream_samples(self):
        loop = asyncio.get_running_loop()

        try:
            while not self.stop_event.is_set():
                try:
                    nrecv = await loop.run_in_executor(
                        None,
                        self.streamer.recv,
                        self.recv_buffer,
                        self.metadata,
                        timedelta(seconds=1),
                    )
                    if nrecv > 0:
                        sample_bytes = self.recv_buffer[:nrecv].tobytes()
                        if len(sample_bytes) != self.buffer_size * 8:
                            print(f"Short read: {len(sample_bytes)} bytes. Skipping")
                            continue
                        topic = b"samples"
                        length = struct.pack("!I", len(sample_bytes))
                        await self.pub_socket.send_multipart(
                            [topic, length, sample_bytes]
                        )
                    else:
                        await asyncio.sleep(0.01)
                except Exception as e:
                    print(f"Error during stream processing: {e}")
        except asyncio.CancelledError:
            print("Stream Samples: Server shutdown requested (Ctrl+C)")
            self.stop_event.set()
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
                    self.usrp.set_rx_gain(self.gain)

                if "center_freq" in message:
                    self.center_freq = float(message["center_freq"])
                    print(f"Setting center frequency to {self.center_freq}")
                    self.usrp.set_rx_freq(self.center_freq)

                if "sample_rate" in message:
                    self.sample_rate = float(message["sample_rate"])
                    self.usrp.set_rx_rate(self.sample_rate)

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

    parser = argparse.ArgumentParser(description="UHD-based receiver for USRP B210")
    parser.add_argument(
        "--host",
        type=str,
        default=config["Network"]["HOST"],
        help="Host to connect to.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(config["Network"]["PORT"]),
        help="Port number to listen on.",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=float(config["Processing"]["SAMPLE_RATE"]),
        help="Sample rate.",
    )
    parser.add_argument(
        "--freq_offset",
        type=float,
        default=float(config["Demodulation"]["FREQ_OFFSET"]),
        help="Frequency offset for signal shifting (in Hz).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=int(config["Processing"]["CHUNK_SIZE"]),
        help="Chunk size for processing samples.",
    )
    parser.add_argument(
        "--center_freq",
        type=float,
        default=float(config["Server"]["CENTER_FREQ"]),
        help="Center frequency.",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=int(config["Server"]["BUFFER_SIZE"]),
        help="Buffer size.",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=float(config["Server"]["GAIN"]),
        help="Gain.",
    )

    args = parser.parse_args()

    receiver = Receiver(args)
    try:
        await asyncio.gather(receiver.stream_samples(), receiver.control_listener())
    except asyncio.CancelledError:
        print("Server shutdown requested (Ctrl+C)")
    finally:
        print("Do anything further if necessary")


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
