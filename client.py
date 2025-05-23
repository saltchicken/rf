import argparse
import configparser
import asyncio
import struct
import numpy as np
import zmq
import zmq.asyncio
from scipy.signal import decimate, firwin, lfilter, resample_poly
from scipy.io.wavfile import write

import pyaudio

config = configparser.ConfigParser()
config.read('config.ini')

HOST = config['Network']['HOST']  # e.g., '127.0.0.1'
PORT = config['Network']['PORT']  # e.g., '5556'
CHUNK_SIZE = 4096
ACCUM_CHUNKS = 16
FFT_SIZE = CHUNK_SIZE * ACCUM_CHUNKS

def fm_demodulate(iq):
    angle = np.angle(iq[1:] * np.conj(iq[:-1]))
    return angle

def complex_from_bytes(data):
    samples = np.frombuffer(data, dtype=np.float32)
    return samples[0::2] + 1j * samples[1::2]

class RingBuffer:
    def __init__(self, size, dtype=np.complex64):
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)
        self.write_ptr = 0
        self.read_ptr = 0
        self.filled = 0

    def append(self, data):
        data_len = len(data)
        if data_len > self.size:
            data = data[-self.size:]  # only keep last `size` elements
            data_len = self.size

        for i in range(data_len):
            self.buffer[self.write_ptr] = data[i]
            self.write_ptr = (self.write_ptr + 1) % self.size
            if self.filled < self.size:
                self.filled += 1
            else:
                self.read_ptr = (self.read_ptr + 1) % self.size

    def read(self, n):
        if self.filled < n:
            return None
        idx = (self.read_ptr + np.arange(n)) % self.size
        result = self.buffer[idx].copy()
        self.read_ptr = (self.read_ptr + n) % self.size
        self.filled -= n
        return result

    def available(self):
        return self.filled

class Reader:
    def __init__(self, sample_rate, freq_offset):
        self.sample_rate = sample_rate
        self.freq_offset = freq_offset

        self.stop_event = asyncio.Event()
        self.sample_queue = asyncio.Queue(maxsize=10)

        self.ctx = zmq.asyncio.Context()

    async def receive_samples(self):
        socket = self.ctx.socket(zmq.SUB)
        socket.connect(f"tcp://{HOST}:{PORT}")
        socket.setsockopt_string(zmq.SUBSCRIBE, '')

        ring = RingBuffer(size=FFT_SIZE * 4)

        try:
            while not self.stop_event.is_set():
                topic, msg = await socket.recv_multipart()
                length = struct.unpack('!I', msg[:4])[0]
                data = msg[4:]

                if len(data) % 8 != 0:
                    print(f"Received invalid buffer of length {len(data)}")
                    continue

                samples = complex_from_bytes(data)
                ring.append(samples)

                while ring.available() >= FFT_SIZE:
                    chunk = ring.read(FFT_SIZE)
                    try:
                        self.sample_queue.put_nowait(chunk)
                    except asyncio.QueueFull:
                        print("Queue full. Dropping frame.")
        except asyncio.CancelledError:
            print("Cancelled receive_samples.")
        finally:
            print("Receiver shutting down.")
            self.stop_event.set()
            socket.close()

    async def record_sample(self, duration_seconds=1):
        total_samples = []
        samples_recorded = 0
        while not self.stop_event.is_set():
            samples = await self.sample_queue.get()
            print(f"Received buffer length: {len(samples)}")
            samples_recorded += len(samples)
            total_samples.append(samples)
            self.sample_queue.task_done()

            if samples_recorded >= self.sample_rate * duration_seconds:
                self.stop_event.set()

        samples = np.concatenate(total_samples, axis=0)
        # freq_offset = 3e5
        t = np.arange(len(samples)) / self.sample_rate
        shifted_samples = samples * np.exp(-1j * 2 * np.pi * self.freq_offset * t)

        # Design low-pass FIR filter (cutoff ~100 kHz)
        cutoff_hz = 100e3
        nyquist_rate = self.sample_rate / 2
        num_taps = 101
        fir_coeff = firwin(num_taps, cutoff_hz / nyquist_rate)

        filtered_samples = lfilter(fir_coeff, 1.0, shifted_samples)
        fm_demod = fm_demodulate(filtered_samples)

        decimation_factor = int(self.sample_rate / 48000)
        audio = resample_poly(fm_demod, up=1, down=decimation_factor)

        audio /= np.max(np.abs(audio))
        audio_int16 = np.int16(audio * 32767)

        write("output.wav", 48000, audio_int16)
        print("Saved FM audio to output.wav")

    async def listen_sample(self):
        # PyAudio setup
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=48000,
                        output=True,
                        frames_per_buffer=1024)

        try:
            while not self.stop_event.is_set():
                samples = await self.sample_queue.get()
                self.sample_queue.task_done()
                print(len(samples))

                # Frequency shift
                t = np.arange(len(samples)) / self.sample_rate
                shifted_samples = samples * np.exp(-1j * 2 * np.pi * self.freq_offset * t)

                # FIR low-pass filter
                cutoff_hz = 100e3
                nyquist_rate = self.sample_rate / 2
                num_taps = 101
                fir_coeff = firwin(num_taps, cutoff_hz / nyquist_rate)

                filtered_samples = lfilter(fir_coeff, 1.0, shifted_samples)

                # FM demodulation
                fm_demod = fm_demodulate(filtered_samples)

                # Decimate to ~48kHz
                decimation_factor = int(self.sample_rate / 48000)
                audio = resample_poly(fm_demod, up=1, down=decimation_factor)

                # Normalize and convert to int16
                audio /= np.max(np.abs(audio) + 1e-9)  # prevent division by zero
                volume = 0.25  # 25% volume
                audio *= volume
                audio_int16 = np.int16(audio * 32767)


                # Stream to audio output
                stream.write(audio_int16.tobytes())
        except asyncio.CancelledError:
            print("Cancelled record_sample.")
        finally:
            print("Shutting down audio playback.")
            stream.stop_stream()
            stream.close()
            p.terminate()
            self.stop_event.set()

async def main():
    parser = argparse.ArgumentParser(description='FM receiver and demodulator.')
    parser.add_argument('--freq_offset', type=float, default=3e5,
                        help='Frequency offset for signal shifting (in Hz). Default is 300000.')
    args = parser.parse_args()

    sample_rate = float(config['Processing']['SAMPLE_RATE'])
    reader = Reader(sample_rate, args.freq_offset)
    consumer_task = asyncio.create_task(reader.record_sample())
    producer_task = asyncio.create_task(reader.receive_samples())
    await asyncio.gather(producer_task, consumer_task)

if __name__ == '__main__':
    asyncio.run(main())

