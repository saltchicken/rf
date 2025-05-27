from multiprocessing import Pool
import asyncio
import numpy as np
from rfanalyze import ReaderRecorder, get_args
from scipy.signal import resample

from pyAudioAnalysis import ShortTermFeatures

import tensorflow as tf
import tensorflow_hub as hub

import pandas as pd

import time
import configparser

from pathlib import Path
config_dir = f'{Path(__file__).parent}/config'


start = time.time()
yamnet_model_handle = "yamnet"
yamnet = hub.load(yamnet_model_handle)
end = time.time()
print(f'Loading model took {end - start} seconds')

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

def analyze_audio(audio_int16, sample_rate=16000):
    audio_float = audio_int16.astype(np.float32) / 32768.0
    scores, embeddings, spectrogram = yamnet(audio_float)

    with open('yamnet/assets/yamnet_class_map.csv', 'r', encoding='utf-8') as f:
        class_map = f.read().splitlines()

# Skip the header and extract the third column (display_name)
    class_names = [line.split(',')[2] for line in class_map[1:]]

    mean_scores = tf.reduce_mean(scores, axis=0)
    top_class = tf.argmax(mean_scores)
    top_score = mean_scores[top_class].numpy()
    top_label = class_names[top_class.numpy()]

    # print(f"Top class: {top_label} (score: {top_score:.3f})")   
    return top_label, top_score




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
        # args.output_filename = f'output_{index:03d}.wav'

        reader_recorder = ReaderRecorder(args)
        settings = await reader_recorder.get_current_settings()
        frequency = int(settings['center_freq']) + int(freq_offset)
        reader_recorder.output_filename = f'output_{str(frequency)}.wav'

        return await reader_recorder.run()

    return asyncio.run(runner())


def main():
    config = configparser.ConfigParser()
    config.read(f'./src/rfanalyze/config/config.ini')

    # TODO: Fix these
    sample_rate = float(config['Processing']['SAMPLE_RATE'])
    center_freq = config['Server']['CENTER_FREQ']
    # channel_width = 1e5
    channel_width = int(sample_rate // 20)
    print(f"Channel width: {channel_width}")
    print(f"Center freq: {center_freq}")
    freq_offsets = [(i * channel_width) - (channel_width // 2) for i in range(20)]
    mid_offset = (len(freq_offsets) - 1) // 2 * channel_width
    centered_freq_offsets = [f - mid_offset for f in freq_offsets]
    print(f"Centered Freq offsets: {centered_freq_offsets}")
    args_list = [(i, f) for i, f in enumerate(centered_freq_offsets)]


    with Pool(processes=len(args_list)) as pool:
        results = pool.map(run_recorder_instance, args_list)

    for index, result in enumerate(results):
        top_label, top_score = analyze_audio(result, sample_rate=16000)
        print(f"Freq: {centered_freq_offsets[index] + int(center_freq)}, Label: {top_label}, Score: {top_score}")
        # feature, names = extract_features(result, sample_rate=16000)
        # is_speech = is_speech_from_features(feature)
        # print(is_speech)

        # mean_features = np.mean(feature, axis=1)
        # for name, value in zip(names, mean_features):
        #     print(f"Process {index} {name}: {value}")

if __name__ == '__main__':
    main()
