import asyncio
import struct
import numpy as np
from scipy.signal import decimate, firwin, lfilter
from scipy.io.wavfile import write

HOST = '127.0.0.1'
PORT = 5000
CHUNK_SIZE = 4096
ACCUM_CHUNKS = 10
FFT_SIZE = CHUNK_SIZE * ACCUM_CHUNKS
SAMPLE_RATE = 10e6

stop_event = asyncio.Event()
sample_queue = asyncio.Queue(maxsize=10)

def fm_demodulate(iq):
    # FM demodulation using phase difference between consecutive IQ samples
    angle = np.angle(iq[1:] * np.conj(iq[:-1]))
    return angle

def complex_from_bytes(data):
    samples = np.frombuffer(data, dtype=np.float32)
    return samples[0::2] + 1j * samples[1::2]

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
                except asyncio.QueueFull:
                    print("Queue full. Dropping frame.")
                    pass

                buffer = buffer[FFT_SIZE:]
                if buffer.shape != (0,):
                    print(f"Buffer shape error: {buffer.shape}")

    except asyncio.IncompleteReadError:
        print("Server closed the connection.")
    except asyncio.CancelledError:
        print("Cancelled receive_samples.")
    finally:
        print("This is closing")
        stop_event.set()

async def print_sample_lengths():
    total_samples = []
    while not stop_event.is_set():
        samples = await sample_queue.get()  # wait for next item
        print(f"Received buffer length: {len(samples)}")
        total_samples.append(samples)
        sample_queue.task_done()

        print(len(total_samples))

        if len(total_samples) >= 2000:
            stop_event.set()

    samples = np.concatenate(total_samples, axis=0)
    freq_offset = 3e5
    t = np.arange(len(samples)) / SAMPLE_RATE
    shifted_samples = samples * np.exp(-1j * 2 * np.pi * freq_offset * t)

    # Design low-pass FIR filter (cutoff ~100 kHz)
    cutoff_hz = 100e3
    nyquist_rate = SAMPLE_RATE / 2
    num_taps = 101  # filter length

    fir_coeff = firwin(num_taps, cutoff_hz / nyquist_rate)

    # Apply low-pass filter to IQ samples to isolate FM channel
    filtered_samples = lfilter(fir_coeff, 1.0, shifted_samples)

    # FM demodulate filtered samples
    fm_demod = fm_demodulate(filtered_samples)

    # Decimate to 48 kHz audio
    decimation_factor = int(SAMPLE_RATE / 48000)
    audio = decimate(fm_demod, decimation_factor)

    # Normalize audio
    audio /= np.max(np.abs(audio))
    audio_int16 = np.int16(audio * 32767)

    # Save audio
    write("output.wav", 48000, audio_int16)
    print("Saved FM audio to output.wav")

async def main():
    consumer_task = asyncio.create_task(print_sample_lengths())
    producer_task = asyncio.create_task(receive_samples())
    await asyncio.gather(producer_task, consumer_task)

if __name__ == '__main__':
    asyncio.run(main())
