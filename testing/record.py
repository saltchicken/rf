import SoapySDR
from SoapySDR import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate, firwin, lfilter
from scipy.io.wavfile import write

def fm_demodulate(iq):
    # FM demodulation using phase difference between consecutive IQ samples
    angle = np.angle(iq[1:] * np.conj(iq[:-1]))
    return angle

def main():
    # Find SDR device
    results = SoapySDR.Device.enumerate()
    if not results:
        print("No SDR devices found.")
        return
    
    args = results[0]
    sdr = SoapySDR.Device(args)

    sample_rate = 1e6   # 20 MHz sample rate
    center_freq = 92e6  # 92 MHz center frequency

    sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
    sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)

    rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    sdr.activateStream(rxStream)

    duration_seconds = 4
    num_samples = int(sample_rate * duration_seconds)

    buff = np.empty(num_samples, np.complex64)

    # Read samples in chunks
    read_ptr = 0
    chunk_size = 4096

    while read_ptr < num_samples:
        to_read = min(chunk_size, num_samples - read_ptr)
        sr = sdr.readStream(rxStream, [buff[read_ptr:read_ptr+to_read]], to_read)
        if sr.ret > 0:
            read_ptr += sr.ret
        else:
            print(f"Error or timeout in readStream: {sr.ret}")
            break

    sdr.deactivateStream(rxStream)
    sdr.closeStream(rxStream)

    if read_ptr != num_samples:
        print(f"Warning: only got {read_ptr} samples out of {num_samples}")

    # Frequency shift by -300 kHz to center FM channel at 0 Hz
    freq_offset = 3e5
    t = np.arange(read_ptr) / sample_rate
    shifted_samples = buff[:read_ptr] * np.exp(-1j * 2 * np.pi * freq_offset * t)

    # Design low-pass FIR filter (cutoff ~100 kHz)
    cutoff_hz = 100e3
    nyquist_rate = sample_rate / 2
    num_taps = 101  # filter length

    fir_coeff = firwin(num_taps, cutoff_hz / nyquist_rate)

    # Apply low-pass filter to IQ samples to isolate FM channel
    filtered_samples = lfilter(fir_coeff, 1.0, shifted_samples)

    # FM demodulate filtered samples
    fm_demod = fm_demodulate(filtered_samples)

    # Decimate to 48 kHz audio
    decimation_factor = int(sample_rate / 48000)
    audio = decimate(fm_demod, decimation_factor)

    # Normalize audio
    audio /= np.max(np.abs(audio))
    audio_int16 = np.int16(audio * 32767)

    # Save audio
    write("output.wav", 48000, audio_int16)
    print("Saved FM audio to output.wav")

    # plot_fft(shifted_samples, center_freq, freq_offset, sample_rate)

def plot_fft(samples, center_freq, freq_offset, sample_rate):
    # Plot FFT of first 4096 shifted samples for visualization
    fft_data = np.fft.fftshift(np.fft.fft(samples[:4096]))
    fft_magnitude = 20 * np.log10(np.abs(fft_data) + 1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(4096, 1/sample_rate))

    plt.figure(figsize=(10,6))
    plt.plot(freqs, fft_magnitude)
    plt.title(f"FFT of frequency shifted samples at {center_freq/1e6} MHz, shifted by {freq_offset/1e3} kHz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
