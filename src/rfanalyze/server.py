import argparse, configparser
import asyncio
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
import struct
import zmq
import zmq.asyncio

from pathlib import Path
config_dir = f'{Path(__file__).parent}/config'

class Receiver:
    def __init__(self, args):
        self.port = args.port
        self.sample_rate = args.sample_rate
        self.center_freq = args.center_freq
        self.buffer_size = args.buffer_size
        self.gain = args.gain

        self.sdr = None
        self.rxStream = None
        # self.setup_sdr(args.driver)
        self.setup_sdr()

        self.stop_event = asyncio.Event()

        self.ctx = zmq.asyncio.Context()
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://0.0.0.0:{self.port}")
        print(f"ZeroMQ PUB server broadcasting on port {self.port}")

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

        # sdr.setAntenna(SOAPY_SDR_RX, 0, "LNAW")
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
        # gain_range = self.sdr.getGainRange(SOAPY_SDR_RX, 0)
        # print(gain_range)
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
                self.ctx.term()
                print("Receiver cleanup finished.")

    async def stream_samples(self):
        self.rxStream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rxStream)
        loop = asyncio.get_running_loop()

        try:
            while not self.stop_event.is_set():
                try:
                    buff = np.empty(self.buffer_size, np.complex64)
                    sr = await loop.run_in_executor(None, self.sdr.readStream, self.rxStream, [buff], self.buffer_size)
                    if sr.ret > 0:
                        sample_bytes = buff[:sr.ret].tobytes()
                        if len(sample_bytes) != self.buffer_size * 8:
                            print(f"Short read: {len(sample_bytes)} bytes. Breaking")
                            break
                        # Send topic + message
                        topic = b"samples"
                        length = struct.pack('!I', len(sample_bytes))
                        await self.pub_socket.send_multipart([topic, length, sample_bytes])
                    else:
                        await asyncio.sleep(0.01)
                except Exception as e:
                    print(f"Error during stream processing: {e}")

        except asyncio.CancelledError:
            print("Server shutdown requested (Ctrl+C)")
            self.stop_event.set()
        finally:
            await asyncio.sleep(1)
            self.close()

async def run():
    config = configparser.ConfigParser()
    config.read(f'{config_dir}/config.ini')

    parser = argparse.ArgumentParser(description='FM receiver and demodulator')
    parser.add_argument('--host', type=str, default=config['Network']['HOST'], help='Host to connect to.')
    parser.add_argument('--port', type=int, default=config['Network']['PORT'], help='Port number to listen on.')
    parser.add_argument('--sample_rate', type=float, default=config['Processing']['SAMPLE_RATE'], help='Sample rate.')
    parser.add_argument('--freq_offset', type=float, default=config['Demodulation']['FREQ_OFFSET'], help='Frequency offset for signal shifting (in Hz).')
    parser.add_argument('--chunk_size', type=int, default=config['Processing']['CHUNK_SIZE'], help='Chunk size for processing samples.')
    parser.add_argument('--center_freq', type=float, default=config['Server']['CENTER_FREQ'], help='Center frequency.')
    parser.add_argument('--buffer_size', type=int, default=config['Server']['BUFFER_SIZE'], help='Buffer size.')
    parser.add_argument('--gain', type=float, default=config['Server']['GAIN'], help='Gain.')

    args = parser.parse_args()

    receiver = Receiver(args)
    await receiver.stream_samples()

def main():
    asyncio.run(run())

if __name__ == '__main__':
    main()
