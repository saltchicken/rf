import asyncio
import argparse, configparser
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
import signal

import zmq
import zmq.asyncio

from client import Reader

plt.style.use('dark.mplstyle')


class RealTimeFFTVisualizer(Reader):
    def __init__(self, args):
        super().__init__(args)
        self.chunk_size = 65536
        self.DECIMATION_FACTOR = 64

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        x = np.linspace(-self.sample_rate / 2, self.sample_rate / 2, self.chunk_size // self.DECIMATION_FACTOR)
        self.line, = self.ax.plot(x, np.zeros(self.chunk_size // self.DECIMATION_FACTOR))

        self.ax.set_xlim(-self.sample_rate / 2, self.sample_rate / 2)
        self.ax.set_ylim(-100, 100)
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Magnitude (dB)')
        self.ax.set_title('Real-time FFT from SDR samples')
        self.ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        xticks = np.linspace(-self.sample_rate / 2, self.sample_rate / 2, 19)
        xtick_labels = [f'{x / 1e6:.1f} MHz' for x in xticks]
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xtick_labels)

    def complex_from_bytes(self, data):
        samples = np.frombuffer(data, dtype=np.float32)
        return samples[0::2] + 1j * samples[1::2]

    def on_key(self, event):
        if event.key == 'q':
            print("Detected 'q' key press. Quitting...")
            self.stop_event.set()

    def run_asyncio_loop(self):
        asyncio.run(self.receive_samples())

    def update_plot(self, frame):
        if not self.sample_queue.empty():
            samples = self.sample_queue.get_nowait()
            fft_result = np.fft.fftshift(np.fft.fft(samples))
            magnitude = 20 * np.log10(np.abs(fft_result) + 1e-12)
            magnitude = magnitude[::self.DECIMATION_FACTOR]
            self.line.set_ydata(magnitude)
        return self.line,

    def run(self):
        async_thread = threading.Thread(target=self.run_asyncio_loop, daemon=True)
        async_thread.start()

        ani = FuncAnimation(self.fig, self.update_plot, interval=0, blit=True, cache_frame_data=False)
        try:
            plt.show()
        finally:
            self.stop_event.set()
            async_thread.join()


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--host', type=str, default=config['Network']['HOST'], help='Host to connect to.')
    parser.add_argument('--port', type=int, default=config['Network']['PORT'], help='Port number to listen on.')
    parser.add_argument('--sample_rate', type=float, default=config['Processing']['SAMPLE_RATE'], help='Sample rate.')
    parser.add_argument('--freq_offset', type=float, default=config['Demodulation']['FREQ_OFFSET'], help='Frequency offset for signal shifting (in Hz).')
    parser.add_argument('--chunk_size', type=int, default=config['Processing']['CHUNK_SIZE'], help='Chunk size for processing samples.')

    args = parser.parse_args()

    visualizer = RealTimeFFTVisualizer(args)
    visualizer.run()
