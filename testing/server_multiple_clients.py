import asyncio
import SoapySDR
from SoapySDR import *
import numpy as np
import struct

SAMPLE_RATE = 1e6
FREQ = 100e6
BUFFER_SIZE = 4096  # samples per buffer
PORT = 5000

# Global SDR variables
sdr = None
rxStream = None
# List to track connected clients
clients = set()

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

async def broadcast_samples():
    """Continuously read samples and broadcast to all connected clients"""
    global clients
    loop = asyncio.get_running_loop()
    
    while True:
        if not clients:  # No clients connected
            await asyncio.sleep(0.1)
            continue
            
        try:
            buff = np.empty(BUFFER_SIZE, np.complex64)
            sr = await loop.run_in_executor(None, sdr.readStream, rxStream, [buff], BUFFER_SIZE)
            
            if sr.ret > 0:
                samples = buff[:sr.ret].tobytes()
                length_prefix = struct.pack('!I', len(samples))
                data_packet = length_prefix + samples
                
                # Send to all clients
                disconnected_clients = set()
                for writer in clients:
                    try:
                        writer.write(data_packet)
                        await writer.drain()
                    except (ConnectionResetError, BrokenPipeError):
                        disconnected_clients.add(writer)
                    except Exception as e:
                        print(f"Error sending to client: {e}")
                        disconnected_clients.add(writer)
                
                # Remove disconnected clients
                for writer in disconnected_clients:
                    clients.remove(writer)
                    print(f"Client disconnected, {len(clients)} remaining")
            else:
                await asyncio.sleep(0.01)
                
        except Exception as e:
            print(f"Error in broadcast loop: {e}")
            await asyncio.sleep(0.1)

async def handle_client(reader, writer):
    global clients
    addr = writer.get_extra_info('peername')
    print(f"Client connected from {addr}, total clients: {len(clients) + 1}")
    
    # Add this client to our set
    clients.add(writer)
    
    try:
        # Just keep the connection open until client disconnects
        while True:
            try:
                # Optional: you can read commands from clients here
                data = await reader.read(100)
                if not data:  # Client closed connection
                    break
            except:
                break
    finally:
        if writer in clients:
            clients.remove(writer)
        try:
            writer.close()
            await writer.wait_closed()
        except:
            pass
        print(f"Client {addr} disconnected, {len(clients)} remaining")

async def main():
    global sdr, rxStream
    sdr, rxStream = setup_sdr()
    sdr.activateStream(rxStream)
    
    # Start the broadcast task
    broadcast_task = asyncio.create_task(broadcast_samples())
    
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
        # Cancel the broadcast task
        broadcast_task.cancel()
        try:
            await broadcast_task
        except asyncio.CancelledError:
            pass
            
        # Clean up SDR when server exits
        if rxStream:
            sdr.deactivateStream(rxStream)
            sdr.closeStream(rxStream)
        print("SDR resources released")

if __name__ == '__main__':
    asyncio.run(main())
