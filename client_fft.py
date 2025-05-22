import asyncio
import struct
import numpy as np
import matplotlib.pyplot as plt
import time

HOST = '127.0.0.1'
PORT = 5000
CHUNK_SIZE = 4096  # samples per chunk
ACCUM_CHUNKS = 100  # number of chunks to accumulate
FFT_SIZE = CHUNK_SIZE * ACCUM_CHUNKS  # total samples before FFT

def complex_from_bytes(data):
    samples = np.frombuffer(data, dtype=np.float32)
    return samples[0::2] + 1j * samples[1::2]

async def receive_samples(reader, line):
    buffer = np.array([], dtype=np.complex64)

    try:
        while True:
            length_bytes = await reader.readexactly(4)
            (length,) = struct.unpack('!I', length_bytes)

            data = await reader.readexactly(length)
            samples = complex_from_bytes(data)

            buffer = np.concatenate((buffer, samples))

            if len(buffer) >= FFT_SIZE:
                fft_result = np.fft.fftshift(np.fft.fft(buffer[:FFT_SIZE]))
                magnitude = 20 * np.log10(np.abs(fft_result) + 1e-12)

                line.set_ydata(magnitude)
                plt.draw()
                plt.pause(0.001)

                buffer = buffer[FFT_SIZE:]  # remove used samples

    except asyncio.IncompleteReadError:
        print("Server closed the connection.")
    except asyncio.CancelledError:
        pass

async def main():
    reader, _ = await asyncio.open_connection(HOST, PORT)

    plt.ion()
    fig, ax = plt.subplots()
    x = np.linspace(-0.5, 0.5, FFT_SIZE)
    line, = ax.plot(x, np.zeros(FFT_SIZE))
    ax.set_ylim(-100, 10)
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Real-time FFT from SDR samples (accumulated)')

    await receive_samples(reader, line)

if __name__ == '__main__':
    asyncio.run(main())
