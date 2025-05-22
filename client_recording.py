import asyncio
import struct
import numpy as np

HOST = '127.0.0.1'
PORT = 5000
CHUNK_SIZE = 4096
ACCUM_CHUNKS = 10
FFT_SIZE = CHUNK_SIZE * ACCUM_CHUNKS
SAMPLE_RATE = 10e6

sample_queue = asyncio.Queue(maxsize=10)

def complex_from_bytes(data):
    samples = np.frombuffer(data, dtype=np.float32)
    return samples[0::2] + 1j * samples[1::2]

async def receive_samples():
    reader, _ = await asyncio.open_connection(HOST, PORT)
    buffer = np.array([], dtype=np.complex64)
    try:
        # Note: Read one frame before looping to "warm up" the reads
        for i in range(2):
            length_bytes = await reader.readexactly(4)
            (length,) = struct.unpack('!I', length_bytes)

            data = await reader.readexactly(length)

        while True:
            length_bytes = await reader.readexactly(4)
            (length,) = struct.unpack('!I', length_bytes)

            data = await reader.readexactly(length)
            samples = complex_from_bytes(data)
            # print(samples.shape)

            buffer = np.concatenate((buffer, samples))

            if len(buffer) >= FFT_SIZE:
                print("yes")
                try:
                    sample_queue.put_nowait(buffer[:FFT_SIZE])
                except asyncio.QueueFull:
                    print("Queue full. Dropping frame.")
                    pass

                buffer = buffer[FFT_SIZE:]
                if buffer.shape != (0,):
                    print(f"Buffer shape error: {buffer.shape}")

    except asyncio.IncompleteReadError:
        print("Server closed the connection.")
    except asyncio.CancelledError:
        print("Cancelled receive_samples.")
    finally:
        print("Done receiving samples.")

async def print_sample_lengths():
    while True:
        samples = await sample_queue.get()  # wait for next item
        print(f"Received buffer length: {len(samples)}")
        sample_queue.task_done()

async def main():
    consumer_task = asyncio.create_task(print_sample_lengths())
    producer_task = asyncio.create_task(receive_samples())
    await asyncio.gather(producer_task, consumer_task)

if __name__ == '__main__':
    asyncio.run(main())
