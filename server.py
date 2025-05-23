import argparse, configparser
import asyncio
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
import struct
import zmq
import zmq.asyncio

class Receiver:
    def __init__(self, args):
        self.port = args.port
        self.sample_rate = args.sample_rate
        self.center_freq = args.center_freq
        self.buffer_size = args.buffer_size

        self.sdr = None
        # self.setup_sdr(args.driver)
        self.setup_sdr()

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
        self.sdr.setGain(SOAPY_SDR_RX, 0, 60)

    async def stream_samples(self):
        rxStream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.sdr.activateStream(rxStream)
        loop = asyncio.get_running_loop()

        try:
            while True:
                try:
                    buff = np.empty(self.buffer_size, np.complex64)
                    sr = await loop.run_in_executor(None, self.sdr.readStream, rxStream, [buff], self.buffer_size)
                    if sr.ret > 0:
                        samples = buff[:sr.ret].tobytes()
                        if len(samples) != self.buffer_size * 8:
                            print(f"Short read: {len(samples)} bytes. Breaking")
                            break
                        # Send topic + message
                        topic = b"samples"
                        length = struct.pack('!I', len(samples))
                        await self.pub_socket.send_multipart([topic, length, samples])
                    else:
                        await asyncio.sleep(0.01)
                except Exception as e:
                    print(f"Error during stream processing: {e}")
                    brea
        except asyncio.CancelledError:
            pass
        finally:
            print("Closing SDR stream.")
            try:
                self.sdr.deactivateStream(rxStream)
                self.sdr.closeStream(rxStream)
            except Exception as e:
                print(f"Error deactivating/closing SDR stream: {e}")

async def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    parser = argparse.ArgumentParser(description='FM receiver and demodulator.')
    parser.add_argument('--port', type=int, default=config['Network']['PORT'], help='Port number to listen on.',)
    parser.add_argument('--sample_rate', type=float, default=config['Processing']['SAMPLE_RATE'], help='Sample rate.',)
    parser.add_argument('--center_freq', type=float, default=config['Server']['CENTER_FREQ'], help='Center frequency.',)
    parser.add_argument('--buffer_size', type=int, default=config['Server']['BUFFER_SIZE'], help='Buffer size.',)

    args = parser.parse_args()

    receiver = Receiver(args)

    try:
        await receiver.stream_samples()
    except KeyboardInterrupt:
        print("Server shutdown requested (Ctrl+C)")
    finally:
        receiver.pub_socket.close()
        receiver.ctx.term()
        print("ZeroMQ resources released")

if __name__ == '__main__':
    asyncio.run(main())

