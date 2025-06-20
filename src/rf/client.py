import asyncio
import math
import struct
import numpy as np
import zmq
import zmq.asyncio
from scipy.signal import decimate, resample_poly, resample
import time
import json

import multiprocessing
from multiprocessing import Process, Queue
import queue

import pyaudio

from .bridge import Publisher

from .classes import Signal, FFT, FM


def complex_from_bytes(data):
    samples = np.frombuffer(data, dtype=np.float32)
    return samples[0::2] + 1j * samples[1::2]


class Controller:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.ctx = zmq.asyncio.Context()
        self.req_socket = self.ctx.socket(zmq.REQ)
        self.req_socket.connect(f"tcp://{self.host}:{self.port}")

    def get_current_settings(self):
        self.req_socket.send_json({"settings": True})
        response = self.req_socket.recv_json()
        return response

    async def update_settings(self, reader):
        print("Updating settings manually")
        settings = await self.get_current_settings()
        if settings.get("sample_rate") is not None:
            reader.sample_rate = settings["sample_rate"]
        else:
            print("THIS WAS NONE WHY1")
        if settings.get("center_freq") is not None:
            reader.center_freq = settings["center_freq"]
        else:
            print("THIS WAS NONE WHY2")
        if settings.get("gain") is not None:
            reader.gain = settings["gain"]
        else:
            print("THIS WAS NONE WHY3")

    async def set_setting(self, setting, value):
        self.req_socket.send_json({setting: value})
        response = await self.req_socket.recv_json()
        if response["status"] != "ok":
            raise Exception(f"Failed to set {setting} to {value}")
        else:
            print(f"Successfully set {setting} to {value}")
            settings = response["settings"]
            self.sample_rate = settings["sample_rate"]
            self.center_freq = settings["center_freq"]
            self.gain = settings["gain"]
            print("Set all the things after modifying the receiver.")
        return response


class Reader:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sample_rate = None
        self.center_freq = None
        self.freq_offset = 100000  # TODO: Update this dynamically on create
        self.chunk_size = 4096

        self.stop_event = asyncio.Event()
        self.sample_queue = asyncio.Queue(maxsize=10)

        self.ctx = zmq.asyncio.Context()

    # @classmethod
    # async def create(cls, host, port):
    #     reader = cls(host, port)
    #     await reader.update_settings()
    #     return reader

    async def receive_samples(self):
        socket = self.ctx.socket(zmq.SUB)
        socket.connect(f"tcp://{self.host}:{self.port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics

        buffer = np.array([], dtype=np.complex64)

        try:
            while not self.stop_event.is_set():
                (
                    topic,
                    length,
                    sample_bytes,
                ) = await socket.recv_multipart()  # Receives one full message
                length = struct.unpack("!I", length)[0]
                if len(sample_bytes) % 8 != 0:
                    print(f"Received invalid buffer of length {len(sample_bytes)}")
                    continue
                samples = complex_from_bytes(sample_bytes)
                buffer = np.concatenate((buffer, samples))

                if len(buffer) >= self.chunk_size:
                    try:
                        self.sample_queue.put_nowait(buffer[: self.chunk_size])
                    except asyncio.QueueFull:
                        pass
                        # print("Queue full. Dropping frame.")

                    buffer = buffer[self.chunk_size :]
                    if buffer.shape != (0,):
                        print(f"Buffer shape error: {buffer.shape}")

        except asyncio.CancelledError:
            print("Cancelled receive_samples.")
        finally:
            print("Receiver shutting down.")
            socket.close()
            self.close()

    def close(self):
        self.stop_event.set()
        self.ctx.term()


class ReaderRecorder(Reader):
    def __init__(self, host, port):
        super().__init__(host, port)
        self.duration_seconds = None
        self.output_filename = None

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

        fm = FM(signal.samples, self.sample_rate)
        fm.apply_fm_demodulation()
        fm.apply_resample(16000)  # This is now audio
        fm.apply_normalization()
        fm.convert_to_int16()

        if self.output_filename is not None:
            fm.save(self.output_filename)

        return fm.samples

    async def run(self):
        record_task = asyncio.create_task(self.record_sample())
        receive_task = asyncio.create_task(self.receive_samples())
        results = await asyncio.gather(record_task, receive_task)
        return results[0]


class ReaderListener(Reader):
    def __init__(self, host, port, publisher_port):
        super().__init__(host, port)
        self.publisher_port = publisher_port
        self.publisher = Publisher(port=self.publisher_port)
        self.previous_magnitude = None
        self.ema_alpha = 0.3
        self.audio_sample_rate = 16000
        self.sample_buffer_len = 160000
        self.audio_frames_per_buffer = math.ceil(
            self.sample_buffer_len
            / (2000000 // self.audio_sample_rate)  # TODO: Replace with self.sample_rate
        )
        print(f"Audio frames per buffer: {self.audio_frames_per_buffer}")
        self.audio_queue = multiprocessing.Queue(maxsize=10)
        self.audio_queue_stop_event = multiprocessing.Event()
        self.audio_proc = Process(
            target=self.audio_process_worker,
            args=(self.audio_queue, self.audio_queue_stop_event),
        )
        self.audio_proc.start()

    def audio_process_worker(self, audio_queue, audio_queue_stop_event):
        def callback(in_data, frame_count, time_info, status):
            if audio_queue_stop_event.is_set():
                return None, pyaudio.paComplete
            try:
                audio_chunk = audio_queue.get_nowait()
            except queue.Empty:
                print("Audio queue empty, inserting silence")
                audio_chunk = b"\x00" * frame_count * 2
            return (audio_chunk, pyaudio.paContinue)

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.audio_sample_rate,
            output=True,
            stream_callback=callback,
            frames_per_buffer=self.audio_frames_per_buffer,
        )
        stream.start_stream()
        try:
            while stream.is_active():
                time.sleep(1)

        except Exception as e:
            print(f"Error in audio_process_worker: {e}")
        finally:
            print("Closing audio_process_worker")
            stream.stop_stream()
            stream.close()
            p.terminate()
            print("Audio process closed")

    async def listen_sample(self):
        try:
            sample_buffer = np.array([], dtype=np.complex64)
            while not self.stop_event.is_set():
                samples = await self.sample_queue.get()
                self.sample_queue.task_done()

                sample_buffer = np.concatenate((sample_buffer, samples))

                if len(sample_buffer) < self.sample_buffer_len:
                    continue

                else:
                    samples = sample_buffer[: self.sample_buffer_len]
                    sample_buffer = sample_buffer[self.sample_buffer_len :]

                    # Frequency shift
                    signal = Signal(samples, self.sample_rate)
                    signal.apply_freq_offset(self.freq_offset)
                    signal.apply_low_pass_filter(100e3)

                    # Process audio
                    fm = FM(signal.samples, self.sample_rate)
                    fm.apply_fm_demodulation()
                    fm.apply_resample(self.audio_sample_rate)
                    fm.apply_de_emphasis_filter()

                    # Create FFT from the demodulated audio signal
                    fft = FFT(np.copy(fm.samples), self.audio_sample_rate)
                    fft.apply_gaussian_filter(2)
                    fft.apply_median_filter(5)
                    fft.apply_smooth_moving_average(5)

                    # Apply EMA smoothing
                    smoothed = fft.apply_exponential_moving_average(
                        self.previous_magnitude
                        if hasattr(self, "previous_magnitude")
                        else None,
                        self.ema_alpha,
                    )
                    self.previous_magnitude = smoothed
                    fft.magnitude = smoothed

                    mid = len(fft.magnitude) // 2
                    fft.magnitude = fft.magnitude[mid:]

                    # Send FFT data
                    # TODO: This is what is hanging. Properly cancel connections from the UI
                    data = fft.magnitude.tobytes()
                    try:
                        self.publisher.queue.put_nowait(data)
                    except asyncio.QueueFull:
                        print("Queue full. Dropping frame.")

                    # Complete audio processing
                    fm.apply_normalization()
                    fm.apply_volume(0.25)
                    fm.convert_to_int16()

                    # Stream to audio output
                    try:
                        self.audio_queue.put_nowait(fm.samples)
                    except queue.Full:
                        print("Audio Queue full. Dropping frame.")

        except asyncio.CancelledError:
            print("Cancelled record_sample.")
        except Exception as e:
            print(f"Error in record_sample: {e}")
        finally:
            self.stop_event.set()
            self.audio_queue_stop_event.set()

    async def run(self):
        receive_task = asyncio.create_task(self.receive_samples())
        listen_task = asyncio.create_task(self.listen_sample())
        results = await asyncio.gather(listen_task, receive_task)
        return results[0]


class ReaderFFT(Reader):
    def __init__(self, host, port, publisher_port):
        super().__init__(host, port)
        self.fft_size = 1024

        self.publisher_port = publisher_port
        self.publisher = Publisher(port=self.publisher_port)
        self.previous_magnitude = None
        self.ema_alpha = 0.3

    async def analyze_sample(self):
        total_samples = []
        samples_recorded = 0

        # # TODO: Take a relook at this. Calling a whole request that may not be necessary and also may need to be updated later
        # settings = await self.get_current_settings()
        # freqs = np.fft.fftshift(
        #     np.fft.fftfreq(self.fft_size, 1 / self.sample_rate)
        # ).astype(np.float32)
        # freqs += float(settings["center_freq"])
        # self.publisher.data = freqs.tolist()

        while not self.stop_event.is_set():
            samples = await self.sample_queue.get()
            samples_recorded += 1
            total_samples.append(samples)
            self.sample_queue.task_done()

            if samples_recorded >= 16:  # 32, 64, 128
                samples_recorded = 0

                samples = np.concatenate(total_samples, axis=0)
                total_samples = []

                signal = Signal(samples, self.sample_rate)
                # fft = FFT(samples, self.sample_rate, freqs)
                fft = FFT(samples, self.sample_rate)
                fft.apply_gaussian_filter(2)
                fft.apply_median_filter(5)
                fft.apply_smooth_moving_average(5)

                smoothed = fft.apply_exponential_moving_average(
                    self.previous_magnitude, self.ema_alpha
                )
                self.previous_magnitude = smoothed
                fft.magnitude = smoothed

                # data = np.concatenate((fft.freqs, fft.magnitude)).tobytes()
                data = fft.magnitude.tobytes()
                try:
                    self.publisher.queue.put_nowait(data)
                except asyncio.QueueFull:
                    print("Queue full. Dropping frame.")

    async def run(self):
        record_task = asyncio.create_task(self.analyze_sample())
        receive_task = asyncio.create_task(self.receive_samples())
        try:
            results = await asyncio.gather(
                record_task, receive_task, self.publisher.server_task
            )
            return results[0]
        except asyncio.CancelledError:
            print("Cancelled run.")
        except Exception as e:
            print(f"Error in run: {e}")


# class ReaderConstellation(Reader):
#     def __init__(self, args):
#         super().__init__(args)
#         self.publisher = Publisher()
#         self.constellation_size = 1024  # Number of I/Q points to send at once
#         self.freq_offset = args.freq_offset
#
#     # def fm_demodulate(self, samples: np.ndarray) -> np.ndarray:
#     #     # FM demodulation using phase difference of IQ samples
#     #     phase_diff = np.angle(samples[1:] * np.conj(samples[:-1]))
#     #
#     #     # Resample to 1024 samples
#     #     resampled_phase_diff = resample(phase_diff, 1024)
#     #
#     #     return resampled_phase_diff
#
#     async def analyze_sample(self):
#         iq_samples = []
#
#         while not self.stop_event.is_set():
#             samples = await self.sample_queue.get()
#             self.sample_queue.task_done()
#
#             iq_samples.append(samples)
#
#             if len(iq_samples) >= 8:
#                 samples = np.concatenate(iq_samples, axis=0)
#
#                 # Frequency shift
#                 t = np.arange(len(samples)) / self.sample_rate
#                 samples = samples * np.exp(-1j * 2 * np.pi * self.freq_offset * t)
#
#                 # samples = frequency_shift(samples, self.freq_offset, self.sample_rate)
#                 iq_samples = []
#
#                 # Remove DC offset
#                 samples -= np.mean(samples)
#
#                 # samples = self.fm_demodulate(samples)
#                 # Normalize or decimate if needed
#                 if len(samples) > self.constellation_size:
#                     step = len(samples) // self.constellation_size
#                     samples = resample_poly(samples, up=1, down=step)
#
#                 # Prepare IQ data for publishing
#                 iq_data = np.vstack((samples.real, samples.imag)).astype(np.float32)
#                 data = iq_data.T.flatten().tobytes()  # [I0, Q0, I1, Q1, ...]
#                 #
#                 # i_part = samples.real.astype(np.float32)
#                 # q_part = samples.imag.astype(np.float32)
#                 # data = np.concatenate((i_part, q_part)).tobytes()
#
#                 self.publisher.publisher.send(data)
#
#     async def run(self):
#         record_task = asyncio.create_task(self.analyze_sample())
#         receive_task = asyncio.create_task(self.receive_samples())
#         results = await asyncio.gather(
#             record_task, receive_task, self.publisher.server_task
#         )
#         return results[0]
