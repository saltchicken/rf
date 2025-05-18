import SoapySDR
from SoapySDR import *
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import firwin, lfilter, decimate

class SDRRecorder:
    def __init__(self, sample_rate=1e6):
        # Find SDR device
        results = SoapySDR.Device.enumerate()
        if not results:
            raise RuntimeError("No SDR devices found.")
        
        self.args = results[0]
        self.sdr = SoapySDR.Device(self.args)
        self.sample_rate = sample_rate
        self.rxStream = None
        
        # Set initial sample rate
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
        
    def record(self, center_freq, duration_seconds=4, freq_offset=0):
        """Record samples from specified frequency for given duration"""
        # Set frequency
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
        
        # Setup stream if not already done
        if self.rxStream is None:
            self.rxStream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        
        # Activate stream
        self.sdr.activateStream(self.rxStream)
        
        # Prepare buffer
        num_samples = int(self.sample_rate * duration_seconds)
        buff = np.empty(num_samples, np.complex64)
        
        # Read samples in chunks
        read_ptr = 0
        chunk_size = 4096
        
        while read_ptr < num_samples:
            to_read = min(chunk_size, num_samples - read_ptr)
            sr = self.sdr.readStream(self.rxStream, [buff[read_ptr:read_ptr+to_read]], to_read)
            if sr.ret > 0:
                read_ptr += sr.ret
                # Calculate and print processing speed every ~1 second
            else:
                print(f"Error or timeout in readStream: {sr.ret}")
                break
        
        # Deactivate stream
        self.sdr.deactivateStream(self.rxStream)
        
        if read_ptr != num_samples:
            print(f"Warning: only got {read_ptr} samples out of {num_samples}")
        
        # Apply frequency shift if needed
        if freq_offset != 0:
            t = np.arange(read_ptr) / self.sample_rate
            buff[:read_ptr] = buff[:read_ptr] * np.exp(-1j * 2 * np.pi * freq_offset * t)
        
        return buff[:read_ptr]
    
    def fm_demodulate_and_save(self, samples, output_file="output.wav", cutoff_hz=100e3):
        """Process FM samples and save to WAV file"""
        # Design low-pass FIR filter
        nyquist_rate = self.sample_rate / 2
        num_taps = 101
        fir_coeff = firwin(num_taps, cutoff_hz / nyquist_rate)
        
        # Apply low-pass filter
        filtered_samples = lfilter(fir_coeff, 1.0, samples)
        
        # FM demodulate
        angle = np.angle(filtered_samples[1:] * np.conj(filtered_samples[:-1]))
        
        # Decimate to 48 kHz audio
        decimation_factor = int(self.sample_rate / 48000)
        audio = decimate(angle, decimation_factor)
        
        # Normalize audio
        audio /= np.max(np.abs(audio))
        audio_int16 = np.int16(audio * 32767)
        
        # Save audio
        write(output_file, 48000, audio_int16)
        print(f"Saved FM audio to {output_file}")
    
    def close(self):
        """Close the SDR device and stream"""
        if self.rxStream is not None:
            self.sdr.closeStream(self.rxStream)
            self.rxStream = None

