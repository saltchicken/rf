import asyncio
import struct
import numpy as np
import zmq
import zmq.asyncio
from scipy.signal import decimate, resample_poly, resample
from scipy.io.wavfile import write
import time

import multiprocessing
from multiprocessing import Process, Queue
import queue

import pyaudio
import wavescope

from .classes import Signal, FFT


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
        self.output_filename = args.output_filename

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
        total_samples = []

        signal = Signal(samples, self.sample_rate)
        signal.apply_freq_offset(self.freq_offset)
        signal.apply_low_pass_filter(100e3)

        fm_demod = fm_demodulate(signal.samples)

        decimation_factor = int(self.sample_rate / 16000)
        audio = resample_poly(fm_demod, up=1, down=decimation_factor)

        audio /= np.max(np.abs(audio))
        audio_int16 = np.int16(audio * 32767)

        if self.output_filename is not None:
            write(self.output_filename, 16000, audio_int16)
            print(f"Saved FM audio to {self.output_filename}")

        return audio_int16

    async def run(self):
        record_task = asyncio.create_task(self.record_sample())
        receive_task = asyncio.create_task(self.receive_samples())
        results = await asyncio.gather(record_task, receive_task)
        return results[0]

class ReaderListener(Reader):
    def __init__(self, args):
        super().__init__(args)
        self.audio_queue = multiprocessing.Queue(maxsize=1000)
        self.audio_queue_stop_event = multiprocessing.Event()
        self.audio_proc = Process(target=self.audio_process_worker, args=(self.audio_queue, self.audio_queue_stop_event))
        self.audio_proc.start()

    def audio_process_worker(self, audio_queue, audio_queue_stop_event):
        def callback(in_data, frame_count, time_info, status):
            if audio_queue_stop_event.is_set():
                return None, pyaudio.paComplete
            try:
                audio_chunk = audio_queue.get_nowait()
            except queue.Empty:
                audio_chunk = b'\x00' * frame_count * 2
            return (audio_chunk, pyaudio.paContinue)

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=48000,
                        output=True,
                        stream_callback=callback,
                        frames_per_buffer=1024
                        )
        stream.start_stream()
        try:
            while stream.is_active():
                time.sleep(0.1)

        except Exception as e:
            print(f"Error in audio_process_worker: {e}")
        finally:
            print("Closing audio_process_worker")
            stream.stop_stream()
            stream.close()
            p.terminate()

    async def listen_sample(self):
        try:
            final = np.array([], dtype=np.int16)
            while not self.stop_event.is_set():
                samples = await self.sample_queue.get()
                self.sample_queue.task_done()

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
                # audio /= np.max(np.abs(audio) + 1e-9)  # prevent division by zero
                max_val = np.max(np.abs(audio))
                if max_val < 1e-9:
                    max_val = 1e-9
                audio /= max_val
                volume = 0.25  # 25% volume
                audio *= volume
                audio_int16 = np.int16(audio * 32767)


                final = np.concatenate((final, audio_int16))


                # Stream to audio output
                if len(final) >= 1024:
                    try:
                        self.audio_queue.put_nowait(final[:1024])
                        final = final[1024:]
                    except queue.Full:
                        print("Audio Queue full. Dropping frame.")
        except asyncio.CancelledError:
            print("Cancelled record_sample.")
        except Exception as e:
            print(f"Error in record_sample: {e}")
        finally:
            self.stop_event.set()
            self.audio_queue_stop_event.set()


def fm_demodulate(iq_samples):
    # Instantaneous phase difference (FM demodulation)
    return np.angle(iq_samples[1:] * np.conj(iq_samples[:-1]))

# def frequency_shift(samples, f_offset, sample_rate):
#     n = np.arange(len(samples))
#     shift_signal = np.exp(1j * 2 * np.pi * f_offset * n / sample_rate)
#     shifted_samples = samples * shift_signal
#     return shifted_samples

class ReaderFFT(Reader):
    def __init__(self, args):
        super().__init__(args)
        self.fft_size = 1024


        self.publisher = wavescope.Publisher()
        self.previous_magnitude = None
        self.ema_alpha = 0.3

    async def analyze_sample(self):
        total_samples = []
        samples_recorded = 0


        freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1/self.sample_rate)).astype(np.float32)
        while not self.stop_event.is_set():
            samples = await self.sample_queue.get()
            samples_recorded += 1
            total_samples.append(samples)
            self.sample_queue.task_done()

            if samples_recorded >= 16: # 32, 64, 128
                samples_recorded = 0

                samples = np.concatenate(total_samples, axis=0)
                total_samples = []

                signal = Signal(samples, self.sample_rate)
                fft = FFT(samples, self.sample_rate, freqs)
                fft.apply_gaussian_filter(2)
                fft.apply_median_filter(5)
                fft.apply_smooth_moving_average(5)

                smoothed = fft.apply_exponential_moving_average(self.previous_magnitude, self.ema_alpha)
                self.previous_magnitude = smoothed
                fft.magnitude = smoothed

                data = np.concatenate((fft.freqs, fft.magnitude)).tobytes()
                self.publisher.publisher.send(data)
                #
                # samples = frequency_shift(samples, 10000, self.sample_rate)
                # Remove DC offset
                # samples -= np.mean(samples) 



                # cutoff_hz = 100e3
                # nyquist_rate = self.sample_rate / 2
                # num_taps = 101
                # fir_coeff = firwin(num_taps, cutoff_hz / nyquist_rate)

                # samples = lfilter(fir_coeff, 1.0, samples)



                # audio = fm_demodulate(samples)
                # audio = resample(audio, len(audio)//8)
                # print(len(audio))

                # if len(samples) > self.fft_size:
                #     step = len(samples) // self.fft_size
                #     samples = resample_poly(samples, up=1, down=step)

                # if len(audio) > 1024 * 16:
                #     step = len(audio) // 1024 * 16
                #     print(step)
                #     audio = resample_poly(audio, up=1, down=step)

                # print(len(audio))

                # freqs = np.linspace(-self.sample_rate/2, self.sample_rate/2, len(audio), endpoint=False).astype(np.float32)
                # Create freqs a count from 0 to self.fft_size
                # freqs = np.linspace(0, self.fft_size, len(audio), endpoint=False).astype(np.float32)
                # audio_data = np.concatenate((freqs, audio))
                # audio_data = audio.astype(np.float32)
                # self.publisher.publisher.send(audio_data.tobytes())



    async def run(self):
        record_task = asyncio.create_task(self.analyze_sample())
        receive_task = asyncio.create_task(self.receive_samples())
        try:
            results = await asyncio.gather(record_task, receive_task, self.publisher.server_task)
            return results[0]
        except asyncio.CancelledError:
            print("Cancelled run.")
        except Exception as e:
            print(f"Error in run: {e}")




class ReaderConstellation(Reader):
    def __init__(self, args):
        super().__init__(args)
        self.publisher = wavescope.Publisher()
        self.constellation_size = 1024  # Number of I/Q points to send at once
        self.freq_offset = args.freq_offset

    # def fm_demodulate(self, samples: np.ndarray) -> np.ndarray:
    #     # FM demodulation using phase difference of IQ samples
    #     phase_diff = np.angle(samples[1:] * np.conj(samples[:-1]))
    #
    #     # Resample to 1024 samples
    #     resampled_phase_diff = resample(phase_diff, 1024)
    #
    #     return resampled_phase_diff

    async def analyze_sample(self):
        iq_samples = []

        while not self.stop_event.is_set():
            samples = await self.sample_queue.get()
            self.sample_queue.task_done()

            iq_samples.append(samples)

            if len(iq_samples) >= 8:
                samples = np.concatenate(iq_samples, axis=0)

                # Frequency shift
                t = np.arange(len(samples)) / self.sample_rate
                samples = samples * np.exp(-1j * 2 * np.pi * self.freq_offset * t)

                # samples = frequency_shift(samples, self.freq_offset, self.sample_rate)
                iq_samples = []

                # Remove DC offset
                samples -= np.mean(samples) 

                # samples = self.fm_demodulate(samples)
                # Normalize or decimate if needed
                if len(samples) > self.constellation_size:
                    step = len(samples) // self.constellation_size
                    samples = resample_poly(samples, up=1, down=step)

                # Prepare IQ data for publishing
                iq_data = np.vstack((samples.real, samples.imag)).astype(np.float32)
                data = iq_data.T.flatten().tobytes()  # [I0, Q0, I1, Q1, ...]
                #
                # i_part = samples.real.astype(np.float32)
                # q_part = samples.imag.astype(np.float32)
                # data = np.concatenate((i_part, q_part)).tobytes()

                self.publisher.publisher.send(data)

    async def run(self):
        record_task = asyncio.create_task(self.analyze_sample())
        receive_task = asyncio.create_task(self.receive_samples())
        results = await asyncio.gather(record_task, receive_task, self.publisher.server_task)
        return results[0]

