import asyncio
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
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

sample_queue = queue.Queue(maxsize=10)
stop_event = threading.Event()

def complex_from_bytes(data):
    samples = np.frombuffer(data, dtype=np.float32)
    return samples[0::2] + 1j * samples[1::2]

def on_key(event):
    if event.key == 'q':
        print("Detected 'q' key press. Quitting...")
        stop_event.set()

async def receive_samples():
    reader, _ = await asyncio.open_connection(HOST, PORT)
    buffer = np.array([], dtype=np.complex64)

    try:
        # Note: Read one frame before looping to "warm up" the reads
        for i in range(2):
            length_bytes = await reader.readexactly(4)
            (length,) = struct.unpack('!I', length_bytes)

            data = await reader.readexactly(length)

        while not stop_event.is_set():
            length_bytes = await reader.readexactly(4)
            (length,) = struct.unpack('!I', length_bytes)

            data = await reader.readexactly(length)
            samples = complex_from_bytes(data)
            # print(samples.shape)

            buffer = np.concatenate((buffer, samples))

            if len(buffer) >= FFT_SIZE:
                try:
                    sample_queue.put_nowait(buffer[:FFT_SIZE])
                except queue.Full:
                    print("Queue full. Dropping frame.")
                    pass

                buffer = buffer[FFT_SIZE:]
                if buffer.shape != (0,):
                    print(f"Buffer shape error: {buffer.shape}")
                # buffer = np.array([], dtype=np.complex64)


    except asyncio.IncompleteReadError:
        print("Server closed the connection.")
    except asyncio.CancelledError:
        print("Cancelled receive_samples.")
    finally:
        stop_event.set()

def run_asyncio_loop():
    asyncio.run(receive_samples())

def update(frame):
    if not sample_queue.empty():
        samples = sample_queue.get_nowait()
        fft_result = np.fft.fftshift(np.fft.fft(samples))
        magnitude = 20 * np.log10(np.abs(fft_result) + 1e-12)
        magnitude = magnitude[::DECIMATION_FACTOR]
        line.set_ydata(magnitude)
    return line,

# Plot setup
fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.mpl_connect('key_press_event', on_key)

x = np.linspace(-SAMPLE_RATE/2, SAMPLE_RATE/2, FFT_SIZE // DECIMATION_FACTOR)
line, = ax.plot(x, np.zeros(FFT_SIZE // DECIMATION_FACTOR))

ax.set_xlim(-SAMPLE_RATE/2, SAMPLE_RATE/2)
ax.set_ylim(-100, 100)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude (dB)')
ax.set_title('Real-time FFT from SDR samples')
ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)


# Add tick labels at the min and max frequency
xticks = np.linspace(-SAMPLE_RATE/2, SAMPLE_RATE/2, 9)  # 9 points from -1 MHz to 1 MHz
xtick_labels = [f'{x/1e6:.1f} MHz' for x in xticks]
# xticks = [-SAMPLE_RATE/2, 0, SAMPLE_RATE/2]
# xtick_labels = [f'{-SAMPLE_RATE/2/1e6:.1f} MHz', '0 Hz', f'{SAMPLE_RATE/2/1e6:.1f} MHz']
ax.set_xticks(xticks)
ax.set_xticklabels(xtick_labels)

# Start the asyncio thread
async_thread = threading.Thread(target=run_asyncio_loop, daemon=True)
async_thread.start()

# Start the animation
ani = FuncAnimation(fig, update, interval=0, blit=True, cache_frame_data=False)

try:
    plt.show()
finally:
    stop_event.set()
    async_thread.join()
