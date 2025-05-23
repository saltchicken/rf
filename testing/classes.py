import SoapySDR
from SoapySDR import *
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import firwin, lfilter, decimate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

plt.style.use('dark.mplstyle')
plt.rcParams['toolbar'] = 'none'

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
        
    def record(self, center_freq, num_samples, freq_offset=0):
        """Record samples from specified frequency for given duration"""
        # Set frequency
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
        
        # Setup stream if not already done
        if self.rxStream is None:
            self.rxStream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        
        # Activate stream
        self.sdr.activateStream(self.rxStream)
        
        # Prepare buffer
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

    def record_duration(self, center_freq, duration_seconds, freq_offset=0):
        """Record samples from specified frequency for given duration"""
        num_samples = int(self.sample_rate * duration_seconds)
        return self.record(center_freq, num_samples, freq_offset)
    
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

    def display_fft(self, sample, sample_rate=None):
        """
        Display the FFT of a recorded audio sample with zero frequency centered.
        
        Parameters:
        -----------
        sample : array_like
            The recorded audio sample
        sample_rate : int
            The sample rate in Hz
        """

        if sample_rate is None:
            sample_rate = self.sample_rate

        # Compute the FFT
        fft_result = np.fft.fft(sample)
        
        # Get the frequencies
        freqs = np.fft.fftfreq(len(sample), 1/sample_rate)
        
        # Shift zero frequency to center
        fft_shifted = np.fft.fftshift(fft_result)
        freqs_shifted = np.fft.fftshift(freqs)
        
        # Plot the centered FFT
        plt.figure(figsize=(10, 6))
        plt.plot(freqs_shifted, np.abs(fft_shifted))
        plt.title('FFT of Recorded Sample (Zero-Centered)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        plt.show()

    def animate_fft(self, center_freq, duration_seconds=30, interval=500, freq_offset=0):
        """
        Create an animated FFT display that updates at regular intervals.
        
        Parameters:
        -----------
        center_freq : float
            Center frequency in Hz
        duration_seconds : int
            Total duration to run the animation
        interval : int
            Update interval in milliseconds
        freq_offset : float
            Frequency offset for shifting
        """
        # Set up the figure and axis
        fig, ax = plt.figure(figsize=(10, 6)), plt.subplot(111)
        plt.title(f'Real-time FFT at {center_freq/1e6:.1f} MHz')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)
        
        # Set frequency
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
        
        # Setup stream if not already done
        if self.rxStream is None:
            self.rxStream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        
        # Activate stream
        self.sdr.activateStream(self.rxStream)
        
        # Number of samples to capture per frame
        frame_samples = int(self.sample_rate * 0.1)  # 100ms of data
        line, = ax.plot([], [])
        
        def init():
            ax.set_xlim(-self.sample_rate/2, self.sample_rate/2)
            ax.set_ylim(0, 1)
            return line,
        
        def update(frame):
            # Capture new data
            buff = np.empty(frame_samples, np.complex64)
            read_ptr = 0
            chunk_size = 4096
            
            while read_ptr < frame_samples:
                to_read = min(chunk_size, frame_samples - read_ptr)
                sr = self.sdr.readStream(self.rxStream, [buff[read_ptr:read_ptr+to_read]], to_read)
                if sr.ret > 0:
                    read_ptr += sr.ret
                else:
                    print(f"Error in readStream: {sr.ret}")
                    break
            
            # Apply frequency shift if needed
            if freq_offset != 0:
                t = np.arange(read_ptr) / self.sample_rate
                buff[:read_ptr] = buff[:read_ptr] * np.exp(-1j * 2 * np.pi * freq_offset * t)
            
            # Compute FFT
            fft_result = np.fft.fft(buff[:read_ptr])
            freqs = np.fft.fftfreq(len(buff[:read_ptr]), 1/self.sample_rate)
            
            # Shift zero frequency to center
            fft_shifted = np.fft.fftshift(fft_result)
            freqs_shifted = np.fft.fftshift(freqs)
            
            # Update plot data
            line.set_data(freqs_shifted, np.abs(fft_shifted))
            
            # Adjust y-axis limit based on current data
            max_val = np.max(np.abs(fft_shifted))
            ax.set_ylim(0, max_val * 1.1)
            
            return line,
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=int(duration_seconds/(interval/1000)),
                            init_func=init, blit=True, interval=interval)
        
        try:
            plt.show()
        finally:
            # Clean up
            self.sdr.deactivateStream(self.rxStream)
    
    def close(self):
        """Close the SDR device and stream"""
        if self.rxStream is not None:
            self.sdr.closeStream(self.rxStream)
            self.rxStream = None

