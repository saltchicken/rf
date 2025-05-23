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

    async def record_sample(self, duration_seconds=1):
        total_samples = []
        samples_recorded = 0
        while not self.stop_event.is_set():
            samples = await self.sample_queue.get()
            # print(f"Received buffer length: {len(samples)}")
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

async def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--host', type=str, default=config['Network']['HOST'], help='Host to connect to.')
    parent_parser.add_argument('--port', type=int, default=config['Network']['PORT'], help='Port number to listen on.')
    parent_parser.add_argument('--sample_rate', type=float, default=config['Processing']['SAMPLE_RATE'], help='Sample rate.')
    parent_parser.add_argument('--freq_offset', type=float, default=config['Demodulation']['FREQ_OFFSET'], help='Frequency offset for signal shifting (in Hz).')
    parent_parser.add_argument('--chunk_size', type=int, default=config['Processing']['CHUNK_SIZE'], help='Chunk size for processing samples.')

    # Main parser
    parser = argparse.ArgumentParser(description='FM receiver and demodulator.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Subparser: listen
    listen_parser = subparsers.add_parser('listen', parents=[parent_parser], help='Listen to FM broadcast')

    # Subparser: record
    record_parser = subparsers.add_parser('record', parents=[parent_parser], help='Record FM to file')
    record_parser.add_argument('--duration', type=int, default=1, help='Recording duration in seconds')

    args = parser.parse_args()

    reader = Reader(args)
    receive_task = asyncio.create_task(reader.receive_samples())

    if args.command == 'record':
        record_task = asyncio.create_task(reader.record_sample(duration_seconds = args.duration))
        await asyncio.gather(record_task, receive_task)
    elif args.command == 'listen':
        listen_task = asyncio.create_task(reader.listen_sample())
        await asyncio.gather(listen_task, receive_task)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == '__main__':
    asyncio.run(main())

