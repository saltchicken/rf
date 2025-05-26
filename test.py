from multiprocessing import Pool
import asyncio
import numpy as np
from rfanalyze import ReaderRecorder, get_args
from scipy.signal import resample

from pyAudioAnalysis import ShortTermFeatures

def downsample_audio(audio_int16, orig_sr=48000, target_sr=16000):
    n_samples = int(len(audio_int16) * target_sr / orig_sr)
    audio_resampled = resample(audio_int16, n_samples).astype(np.int16)
    return audio_resampled

def extract_features(audio_int16, sample_rate=16000):
    # Convert to float32 in [-1, 1]
    # downsampled = downsample_audio(audio_int16, orig_sr=48000, target_sr=16000)
    # audio_float = downsampled.astype(np.float32) / 32768.0
    #
    audio_float = audio_int16.astype(np.float32) / 32768.0

    win_size = int(0.050 * sample_rate)  # 50 ms
    step_size = int(0.025 * sample_rate)  # 25 ms

    features, feature_names = ShortTermFeatures.feature_extraction(
        audio_float, sample_rate, win_size, step_size
    )

    return features, feature_names


def is_speech_from_features(features):
    mean_feats = np.mean(features, axis=1)

    zcr = mean_feats[0]
    energy = mean_feats[1]
    spec_centroid = mean_feats[4]
    spec_entropy = mean_feats[6]

    print(f"ZCR: {zcr:.3f}, Energy: {energy:.3f}, Centroid: {spec_centroid:.1f}, Entropy: {spec_entropy:.3f}")

    # Heuristic thresholds (tweak as needed)
    if (0.02 < zcr < 0.2 and
        energy > 0.01 and
        300 < spec_centroid < 2000 and
        spec_entropy < 0.85):
        return True  # Likely speech
    else:
        return False  # Likely static or noise

def run_recorder_instance(index_freq):
    index, freq_offset = index_freq

    async def runner():
        args = get_args('record')
        args.index = index
        args.freq_offset = freq_offset
        args.output_filename = f'output_{index}.wav'

        reader_recorder = ReaderRecorder(args)
        return await reader_recorder.run()

    return asyncio.run(runner())


def main():
    freq_offsets = [1e5, 1.5e5, 2e5, 3.5e5, 4e5, 4.5e5]
    args_list = [(i, f) for i, f in enumerate(freq_offsets)]

    with Pool(processes=len(args_list)) as pool:
        results = pool.map(run_recorder_instance, args_list)

    for index, result in enumerate(results):
        feature, names = extract_features(result, sample_rate=16000)
        is_speech = is_speech_from_features(feature)
        print(is_speech)

        # mean_features = np.mean(feature, axis=1)
        # for name, value in zip(names, mean_features):
        #     print(f"Process {index} {name}: {value}")

if __name__ == '__main__':
    main()
