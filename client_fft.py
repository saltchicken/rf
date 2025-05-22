import asyncio
import struct
import numpy as np
import matplotlib.pyplot as plt
import time
import signal

plt.style.use('dark.mplstyle')
plt.rcParams['toolbar'] = 'none'

HOST = '127.0.0.1'
PORT = 5000
CHUNK_SIZE = 4096
ACCUM_CHUNKS = 100
FFT_SIZE = CHUNK_SIZE * ACCUM_CHUNKS
SAMPLE_RATE = 2e6

stop_event = asyncio.Event()  # Global stop signal

def complex_from_bytes(data):
    samples = np.frombuffer(data, dtype=np.float32)
    return samples[0::2] + 1j * samples[1::2]

def on_key(event):
    if event.key == 'q':
        print("Detected 'q' key press. Quitting...")
        stop_event.set()  # Trigger the stop event

async def receive_samples(reader, line):
    buffer = np.array([], dtype=np.complex64)

    try:
        while not stop_event.is_set():
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
        print("Cancelled receive_samples.")
    finally:
        plt.close()

async def main():
    reader, _ = await asyncio.open_connection(HOST, PORT)

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.mpl_connect('key_press_event', on_key)

    x = np.linspace(-SAMPLE_RATE/2, SAMPLE_RATE/2, FFT_SIZE)
    line, = ax.plot(x, np.zeros(FFT_SIZE))
    ax.set_xlim(-SAMPLE_RATE/2, SAMPLE_RATE/2)
    ax.set_ylim(-10, 100)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Real-time FFT from SDR samples')

    await receive_samples(reader, line)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user.")
