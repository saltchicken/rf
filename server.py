import configparser
import asyncio
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
import struct
import zmq
import zmq.asyncio

# Load config
config = configparser.ConfigParser()
config.read('config.ini')

SAMPLE_RATE = float(config['Processing']['SAMPLE_RATE'])
FREQ = 106e6
BUFFER_SIZE = 4096  # samples per buffer
ZMQ_PORT = config['Network']['PORT']

# Global SDR variables
sdr = None
rxStream = None

def setup_sdr():
    results = SoapySDR.Device.enumerate()
    if not results:
        raise RuntimeError("No SDR devices found.")
    
    args = results[0]
    sdr = SoapySDR.Device(args)
    sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
    sdr.setFrequency(SOAPY_SDR_RX, 0, FREQ)
    gain_range = sdr.getGainRange(SOAPY_SDR_RX, 0)
    print(gain_range)
    sdr.setGain(SOAPY_SDR_RX, 0, 60)
    return sdr

async def stream_samples(pub_socket, sdr, rxStream):
    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rxStream)
    loop = asyncio.get_running_loop()

    try:
        while True:
            try:
                buff = np.empty(BUFFER_SIZE, np.complex64)
                sr = await loop.run_in_executor(None, sdr.readStream, rxStream, [buff], BUFFER_SIZE)
                if sr.ret > 0:
                    samples = buff[:sr.ret].tobytes()
                    if len(samples) != BUFFER_SIZE * 8:
                        print(f"Short read: {len(samples)} bytes. Breaking")
                        break
                    # Send topic + message
                    topic = b"samples"
                    length_prefix = struct.pack('!I', len(samples))
                    await pub_socket.send_multipart([topic, length_prefix + samples])
                else:
                    await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error during stream processing: {e}")
                break
    except asyncio.CancelledError:
        pass
    finally:
        print("Closing SDR stream.")
        try:
            sdr.deactivateStream(rxStream)
            sdr.closeStream(rxStream)
        except Exception as e:
            print(f"Error deactivating/closing SDR stream: {e}")

async def main():
    global sdr
    sdr = setup_sdr()

    context = zmq.asyncio.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(f"tcp://0.0.0.0:{ZMQ_PORT}")
    print(f"ZeroMQ PUB server broadcasting on port {ZMQ_PORT}")

    try:
        await stream_samples(pub_socket, sdr, rxStream)
    except KeyboardInterrupt:
        print("Server shutdown requested (Ctrl+C)")
    finally:
        pub_socket.close()
        context.term()
        print("ZeroMQ resources released")

if __name__ == '__main__':
    asyncio.run(main())

