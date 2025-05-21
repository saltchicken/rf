import asyncio
import struct
import time

HOST = '127.0.0.1'
PORT = 5000

async def receive_samples(reader):
    try:
        while True:
            length_bytes = await reader.readexactly(4)
            (length,) = struct.unpack('!I', length_bytes)

            data = await reader.readexactly(length)
            print(f"Received {len(data)} bytes ({len(data)//8} complex64 samples) - {time.time()}")
    except asyncio.IncompleteReadError:
        print("Server closed the connection.")
    except asyncio.CancelledError:
        pass

async def main():
    reader, _ = await asyncio.open_connection(HOST, PORT)
    await receive_samples(reader)

if __name__ == '__main__':
    asyncio.run(main())
