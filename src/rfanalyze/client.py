import asyncio
import struct
import numpy as np
import zmq
import zmq.asyncio
from scipy.signal import decimate, firwin, lfilter, resample_poly
from scipy.io.wavfile import write

import pyaudio


def fm_demodulate(iq):
    angle = np.angle(iq[1:] * np.conj(iq[:-1]))
    return angle

def complex_from_bytes(data):
    samples = np.frombuffer(data, dtype=np.float32)
    return samples[0::2] + 1j * samples[1::2]

class Reader:
    def __init__(self, args):
        self.host = args.host
        self.port = args.port
        self.sample_rate = args.sample_rate
        self.freq_offset = args.freq_offset
        self.chunk_size = args.chunk_size

        self.stop_event = asyncio.Event()
        self.sample_queue = asyncio.Queue(maxsize=10)

        self.ctx = zmq.asyncio.Context()

    async def receive_samples(self):
        socket = self.ctx.socket(zmq.SUB)
        socket.connect(f"tcp://{self.host}:{self.port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe to all topics

        buffer = np.array([], dtype=np.complex64)

        try:
            while not self.stop_event.is_set():
                topic, length, sample_bytes = await socket.recv_multipart()  # Receives one full message
                length = struct.unpack('!I', length)[0]
                if len(sample_bytes) % 8 != 0:
                    print(f"Received invalid buffer of length {len(sample_bytes)}")
                    continue
                samples = complex_from_bytes(sample_bytes)
                buffer = np.concatenate((buffer, samples))

                if len(buffer) >= self.chunk_size:
                    try:
                        self.sample_queue.put_nowait(buffer[:self.chunk_size])
                    except asyncio.QueueFull:
                        print("Queue full. Dropping frame.")

                    buffer = buffer[self.chunk_size:]
                    if buffer.shape != (0,):
                        print(f"Buffer shape error: {buffer.shape}")

        except asyncio.CancelledError:
            print("Cancelled receive_samples.")
        finally:
            print("Receiver shutting down.")
            self.stop_event.set()
            socket.close()

class ReaderRecorder(Reader):
    def __init__(self, args):
        super().__init__(args)
        self.duration_seconds = args.duration

    async def record_sample(self):
        total_samples = []
        samples_recorded = 0
        while not self.stop_event.is_set():
            samples = await self.sample_queue.get()
            # print(f"Received buffer length: {len(samples)}")
            samples_recorded += len(samples)
            total_samples.append(samples)
            self.sample_queue.task_done()

            if samples_recorded >= self.sample_rate * self.duration_seconds:
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

        return audio_int16

    async def run(self):
        record_task = asyncio.create_task(self.record_sample())
        receive_task = asyncio.create_task(self.receive_samples())
        results = await asyncio.gather(record_task, receive_task)
        return results[0]

class ReaderListener(Reader):
    def __init__(self, args):
        super().__init__(args)

    async def listen_sample(self):
        # PyAudio setup
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=48000,
                        output=True,
                        frames_per_buffer=4096)

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
