import SoapySDR
from SoapySDR import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import decimate, firwin, lfilter
from scipy.signal.windows import hann
from scipy.io.wavfile import write
import pyaudio
import time
import threading
import queue

def fm_demodulate(iq):
    # FM demodulation using phase difference between consecutive IQ samples
    angle = np.angle(iq[1:] * np.conj(iq[:-1]))
    return angle

def audio_worker(audio_queue, stop_event, audio_rate):
    """Worker thread to process and play audio samples"""
    # Setup PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=audio_rate,
                    output=True)
    
    # Audio smoothing setup
    audio_overlap = 128 + 64
    previous_audio = np.zeros(audio_overlap, dtype=np.float32)
    window = hann(audio_overlap * 2)
    fade_in = window[:audio_overlap]
    fade_out = window[audio_overlap:]
    
    try:
        while not stop_event.is_set():
            try:
                # Get audio data from queue with timeout
                audio = audio_queue.get(timeout=0.1)
                
                if len(audio) > 0:
                    # Apply a bandpass filter to remove very low and high frequencies
                    b, a = signal.butter(4, [300/audio_rate*2, 15000/audio_rate*2], 'bandpass')
                    audio = signal.filtfilt(b, a, audio)
                    
                    # Normalize with dynamic range compression
                    audio = audio / max(0.001, np.max(np.abs(audio)))
                    # Simple compression
                    audio = np.sign(audio) * np.power(np.abs(audio), 0.8)
                    
                    # Apply crossfade with previous chunk
                    if len(audio) > audio_overlap:
                        audio[:audio_overlap] = audio[:audio_overlap] * fade_in + previous_audio * fade_out
                        previous_audio = audio[-audio_overlap:].copy()
                        stream.write(audio.astype(np.float32).tobytes())
                
                audio_queue.task_done()
            except queue.Empty:
                continue
    finally:
        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Audio thread stopped")

def main():
    # Find SDR device
    results = SoapySDR.Device.enumerate()
    if not results:
        print("No SDR devices found.")
        return
    
    args = results[0]
    sdr = SoapySDR.Device(args)

    sample_rate = 2e6   # 2 MHz sample rate
    center_freq = 92e6  # 92 MHz center frequency
    audio_rate = 48000  # Output audio rate

    sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
    sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)

    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rxStream)

    # Improved filter settings
    cutoff_hz = 200e3  # Narrower filter for cleaner audio
    nyquist_rate = sample_rate / 2
    num_taps = 101    # More taps for sharper filter
    fir_coeff = firwin(num_taps, cutoff_hz / nyquist_rate)
    
    # Setup for processing
    chunk_size = 8192 * 16
    decimation_factor = int(sample_rate / audio_rate)
    freq_offset = 3e5
    
    # Buffer for overlap between chunks (to handle filter edge effects)
    overlap = num_taps - 1
    previous_samples = np.zeros(overlap, dtype=np.complex64)
    
    # Create queue and start audio thread
    audio_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory issues
    stop_event = threading.Event()
    audio_thread = threading.Thread(
        target=audio_worker, 
        args=(audio_queue, stop_event, audio_rate)
    )
    audio_thread.daemon = True
    audio_thread.start()
    
    try:
        print("Streaming FM radio. Press Ctrl+C to stop.")
        while True:
            # Allocate buffer for this chunk
            buff = np.zeros(chunk_size, dtype=np.complex64)
            
            # Read samples
            sr = sdr.readStream(rxStream, [buff], chunk_size, timeoutUs=1000000)
            if sr.ret <= 0:
                print(f"Error in readStream: {sr.ret}")
                continue
                
            # Frequency shift
            t = np.arange(sr.ret) / sample_rate
            shifted_samples = buff[:sr.ret] * np.exp(-1j * 2 * np.pi * freq_offset * t)
            
            # Concatenate with previous samples for filtering
            to_filter = np.concatenate([previous_samples, shifted_samples])
            
            # Apply filter
            filtered = lfilter(fir_coeff, 1.0, to_filter)
            
            # Save overlap for next iteration
            previous_samples = shifted_samples[-overlap:]
            
            # Use only the valid part after filtering
            valid_filtered = filtered[overlap:]
            
            # FM demodulate
            demod = fm_demodulate(valid_filtered)
            
            # Decimate to audio rate
            demod = decimate(demod, decimation_factor)
            
            # Apply de-emphasis filter (standard for FM broadcast)
            # FM broadcast uses 75μs in US, 50μs in Europe
            alpha = np.exp(-1.0/(audio_rate * 75e-6))
            audio = np.zeros_like(demod)
            audio[0] = demod[0]
            for i in range(1, len(demod)):
                audio[i] = demod[i] + alpha * audio[i-1]
            
            # Put audio data in queue for audio thread to process
            try:
                # Use non-blocking put with a short timeout
                audio_queue.put(audio, block=False, timeout=0.1)
            except queue.Full:
                # If queue is full, skip this chunk
                print("Audio queue full, skipping chunk")
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Signal audio thread to stop and wait for it
        stop_event.set()
        audio_thread.join(timeout=1.0)
        
        # Cleanup SDR
        sdr.deactivateStream(rxStream)
        sdr.closeStream(rxStream)
        print("Streaming stopped")

if __name__ == "__main__":
    main()
