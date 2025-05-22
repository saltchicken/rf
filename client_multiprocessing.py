import asyncio
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import multiprocessing
import signal

plt.style.use('dark.mplstyle')
plt.rcParams['toolbar'] = 'none'

HOST = '127.0.0.1'
PORT = 5000
CHUNK_SIZE = 4096
ACCUM_CHUNKS = 10
FFT_SIZE = CHUNK_SIZE * ACCUM_CHUNKS
SAMPLE_RATE = 10e6
DECIMATION_FACTOR = 64

def complex_from_bytes(data):
    samples = np.frombuffer(data, dtype=np.float32)
    return samples[0::2] + 1j * samples[1::2]

def on_key(event, stop_event):
    if event.key == 'q':
        print("Detected 'q' key press. Quitting...")
        stop_event.set()
        plt.close()

async def receive_samples(sample_queue, stop_event):
    reader, _ = await asyncio.open_connection(HOST, PORT)
    buffer = np.array([], dtype=np.complex64)

    try:
        for i in range(2):
            length_bytes = await reader.readexactly(4)
            (length,) = struct.unpack('!I', length_bytes)
            data = await reader.readexactly(length)

        while not stop_event.is_set():
            length_bytes = await reader.readexactly(4)
            (length,) = struct.unpack('!I', length_bytes)
            if length != 32768:
                print("Why--------")
            data = await reader.readexactly(length)
            samples = complex_from_bytes(data)
            buffer = np.concatenate((buffer, samples))

            if len(buffer) >= FFT_SIZE:
                try:
                    sample_queue.put_nowait(buffer[:FFT_SIZE])
                except:
                    print("Queue full. Dropping frame.")
                buffer = buffer[FFT_SIZE:]
                # if buffer.shape != (0,):
                #     print(f"Buffer shape error: {buffer.shape}")
                # buffer = np.array([], dtype=np.complex64)

    except asyncio.IncompleteReadError:
        print("Server closed the connection.")
    except asyncio.CancelledError:
        print("Cancelled receive_samples.")
    finally:
        stop_event.set()

def run_asyncio_process(sample_queue, stop_event):
    asyncio.run(receive_samples(sample_queue, stop_event))

def start_asyncio_process(sample_queue, stop_event):
    p = multiprocessing.Process(target=run_asyncio_process, args=(sample_queue, stop_event), daemon=False)
    p.start()
    return p

def main():
    # Setup multiprocessing structures
    sample_queue = multiprocessing.Queue(maxsize=10)
    stop_event = multiprocessing.Event()

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(-SAMPLE_RATE/2, SAMPLE_RATE/2, FFT_SIZE // DECIMATION_FACTOR)
    line, = ax.plot(x, np.zeros(FFT_SIZE // DECIMATION_FACTOR))

    def update(frame):
        if not sample_queue.empty():
            samples = sample_queue.get_nowait()
            print(samples.shape)
            fft_result = np.fft.fftshift(np.fft.fft(samples))
            magnitude = 20 * np.log10(np.abs(fft_result) + 1e-12)
            magnitude = magnitude[::DECIMATION_FACTOR]
            line.set_ydata(magnitude)
        return line,

    def handle_key(event):
        on_key(event, stop_event)

    fig.canvas.mpl_connect('key_press_event', handle_key)

    ax.set_xlim(-SAMPLE_RATE/2, SAMPLE_RATE/2)
    ax.set_ylim(-100, 100)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Real-time FFT from SDR samples')
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    xticks = np.linspace(-SAMPLE_RATE/2, SAMPLE_RATE/2, 9)
    xtick_labels = [f'{x/1e6:.1f} MHz' for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    # Start async process
    proc = start_asyncio_process(sample_queue, stop_event)

    ani = FuncAnimation(fig, update, interval=0, blit=True, cache_frame_data=False)

    try:
        plt.show()
    finally:
        stop_event.set()
        proc.join()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # Important for compatibility on Windows/macOS
    main()
