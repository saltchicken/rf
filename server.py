import asyncio
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np
import struct

SAMPLE_RATE = 10e6
FREQ = 100e6
BUFFER_SIZE = 4096  # samples per buffer
PORT = 5000

# Global SDR variables
sdr = None
rxStream = None

# SDR setup (blocking part)
def setup_sdr():
    results = SoapySDR.Device.enumerate()
    if not results:
        raise RuntimeError("No SDR devices found.")
    
    args = results[0]
    sdr = SoapySDR.Device(args)
    sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE)
    sdr.setFrequency(SOAPY_SDR_RX, 0, FREQ)
    sdr.setGain(SOAPY_SDR_RX, 0, 30)
    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    return sdr, rxStream

async def stream_samples(writer, sdr, rxStream):
    sdr.activateStream(rxStream)
    loop = asyncio.get_running_loop()
    try:
        while True:
            try:
                buff = np.empty(BUFFER_SIZE, np.complex64)
                sr = await loop.run_in_executor(None, sdr.readStream, rxStream, [buff], BUFFER_SIZE)
                if sr.ret > 0:
                    samples = buff[:sr.ret].tobytes()
                    length_prefix = struct.pack('!I', len(samples))
                    try:
                        writer.write(length_prefix + samples)
                        await writer.drain()
                    except (ConnectionResetError, BrokenPipeError) as e:
                        print(f"Client disconnected: {e.__class__.__name__}")
                        break
                    except Exception as e:
                        print(f"Error sending data: {e}")
                        break
                else:
                    await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error during stream processing: {e}")
                break
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print("Server shutdown requested (Ctrl+C)")
    finally:
        print("Closing SDR stream.")
        try:
            sdr.deactivateStream(rxStream)
        except Exception as e:
            print(f"Error deactivating SDR stream: {e}")
        try:
            writer.close()
            await writer.wait_closed()
        except (ConnectionResetError, BrokenPipeError):
            print("Connection already closed by client")
        except Exception as e:
            print(f"Error during cleanup: {e}")

async def handle_client(reader, writer):
    print(f"Client connected from {writer.get_extra_info('peername')}")
    await stream_samples(writer, sdr, rxStream)

async def main():
    global sdr, rxStream
    sdr, rxStream = setup_sdr()
    
    server = await asyncio.start_server(handle_client, host='0.0.0.0', port=PORT)
    print(f"Async SDR server listening on port {PORT}")
    try:
        async with server:
            await server.serve_forever()
    except KeyboardInterrupt:
        print("Server shutdown requested (Ctrl+C)")
    except asyncio.CancelledError:
        print("Server task cancelled")
    finally:
        # Clean up SDR when server exits
        if rxStream:
            sdr.deactivateStream(rxStream)
            sdr.closeStream(rxStream)
        print("SDR resources released")

if __name__ == '__main__':
    asyncio.run(main())
