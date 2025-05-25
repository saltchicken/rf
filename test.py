from multiprocessing import Pool
import asyncio
import numpy as np
from rfanalyze import ReaderRecorder, get_args

import webrtcvad

def root_mean_square(audio_int16):
    # Convert to float
    audio = audio_int16.astype(np.float32)
    
    # Compute RMS (root mean square) energy
    rms = np.sqrt(np.mean(audio**2))
    
    # If RMS energy above threshold, likely voice
    return rms

def zero_crossing_rate(signal):
    zero_crossings = np.sum(np.abs(np.diff(np.sign(signal)))) / 2
    return zero_crossings / len(signal)

def vad_detect(audio_int16, sample_rate=48000):
    vad = webrtcvad.Vad(2)  # 0=less aggressive, 3=more aggressive
    frame_duration = 30  # ms
    frame_bytes = int(sample_rate * frame_duration / 1000) * 2  # 2 bytes per sample
    audio_bytes = audio_int16.tobytes()

    voiced = 0
    total = 0

    for i in range(0, len(audio_bytes) - frame_bytes + 1, frame_bytes):
        frame = audio_bytes[i:i + frame_bytes]
        if vad.is_speech(frame, sample_rate):
            voiced += 1
        total += 1

    ratio = voiced / total if total else 0
    return ratio

def run_recorder_instance(index_freq):
    index, freq_offset = index_freq

    async def runner():
        args = get_args('record')
        args.index = index
        args.freq_offset = freq_offset
        # args.output_filename = f'output_{index}.wav'

        reader_recorder = ReaderRecorder(args)
        return await reader_recorder.run()

    return asyncio.run(runner())


def main():
    freq_offsets = [2e5, 3e5, 4e5, 5e5, 6e5, 7e5]
    args_list = [(i, f) for i, f in enumerate(freq_offsets)]

    with Pool(processes=len(args_list)) as pool:
        results = pool.map(run_recorder_instance, args_list)

    for index, result in enumerate(results):
        zcr = zero_crossing_rate(result)
        print(f"Process {index} zero_crossing_rate: {zcr} root_mean_square {root_mean_square(result)}, voice_ratio: {vad_detect(result)}")


if __name__ == '__main__':
    main()
