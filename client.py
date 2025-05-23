import configparser
import asyncio
import struct
import numpy as np
import zmq
import zmq.asyncio
from scipy.signal import decimate, firwin, lfilter
from scipy.io.wavfile import write

config = configparser.ConfigParser()
config.read('config.ini')

HOST = config['Network']['HOST']  # e.g., '127.0.0.1'
PORT = config['Network']['PORT']  # e.g., '5556'
CHUNK_SIZE = 4096
ACCUM_CHUNKS = 10
FFT_SIZE = CHUNK_SIZE * ACCUM_CHUNKS
SAMPLE_RATE = float(config['Processing']['SAMPLE_RATE'])

stop_event = asyncio.Event()
sample_queue = asyncio.Queue(maxsize=10)

ctx = zmq.asyncio.Context()

def fm_demodulate(iq):
    angle = np.angle(iq[1:] * np.conj(iq[:-1]))
    return angle

def complex_from_bytes(data):
    samples = np.frombuffer(data, dtype=np.float32)
    return samples[0::2] + 1j * samples[1::2]

async def receive_samples():
    socket = ctx.socket(zmq.SUB)
    socket.connect(f"tcp://{HOST}:{PORT}")
    socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe to all topics

    buffer = np.array([], dtype=np.complex64)

    try:
        while not stop_event.is_set():
            topic, msg = await socket.recv_multipart()  # Receives one full message
            length = struct.unpack('!I', msg[:4])[0]
            data = msg[4:]
            if len(data) % 8 != 0:
                print(f"Received invalid buffer of length {len(data)}")
                continue
            samples = complex_from_bytes(data)
            buffer = np.concatenate((buffer, samples))

            if len(buffer) >= FFT_SIZE:
                try:
                    sample_queue.put_nowait(buffer[:FFT_SIZE])
                except asyncio.QueueFull:
                    print("Queue full. Dropping frame.")

                buffer = buffer[FFT_SIZE:]
                if buffer.shape != (0,):
                    print(f"Buffer shape error: {buffer.shape}")

    except asyncio.CancelledError:
        print("Cancelled receive_samples.")
    finally:
        print("Receiver shutting down.")
        stop_event.set()
        socket.close()

async def print_sample_lengths():
    total_samples = []
    while not stop_event.is_set():
        samples = await sample_queue.get()
        print(f"Received buffer length: {len(samples)}")
        total_samples.append(samples)
        sample_queue.task_done()
        print(len(total_samples))

        if len(total_samples) >= 500:
            stop_event.set()

    samples = np.concatenate(total_samples, axis=0)
    freq_offset = 3e5
    t = np.arange(len(samples)) / SAMPLE_RATE
    shifted_samples = samples * np.exp(-1j * 2 * np.pi * freq_offset * t)

    # Design low-pass FIR filter (cutoff ~100 kHz)
    cutoff_hz = 100e3
    nyquist_rate = SAMPLE_RATE / 2
    num_taps = 101
    fir_coeff = firwin(num_taps, cutoff_hz / nyquist_rate)

    filtered_samples = lfilter(fir_coeff, 1.0, shifted_samples)
    fm_demod = fm_demodulate(filtered_samples)

    decimation_factor = int(SAMPLE_RATE / 48000)
    audio = decimate(fm_demod, decimation_factor)

    audio /= np.max(np.abs(audio))
    audio_int16 = np.int16(audio * 32767)

    write("output.wav", 48000, audio_int16)
    print("Saved FM audio to output.wav")

async def main():
    consumer_task = asyncio.create_task(print_sample_lengths())
    producer_task = asyncio.create_task(receive_samples())
    await asyncio.gather(producer_task, consumer_task)

if __name__ == '__main__':
    asyncio.run(main())

