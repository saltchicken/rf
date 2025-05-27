import numpy as np
from scipy.signal import firwin, lfilter, medfilt
from scipy.ndimage import gaussian_filter1d

class Signal:
    def __init__(self, samples, sample_rate):
        self.samples = samples
        self.sample_rate = sample_rate

    def apply_freq_offset(self, freq_offset):
        t = np.arange(len(self.samples)) / self.sample_rate
        self.samples = self.samples * np.exp(-1j * 2 * np.pi * freq_offset * t)

    def apply_low_pass_filter(self, cutoff_hz):
        nyquist_rate = self.sample_rate / 2
        num_taps = 101
        fir_coeff = firwin(num_taps, cutoff_hz / nyquist_rate)
        self.samples = lfilter(fir_coeff, 1.0, self.samples)

class FFT:
    def __init__(self, samples, sample_rate, freqs=None):
        self.samples = samples
        self.sample_rate = sample_rate
        self.fft_size = 1024

        if freqs is None:
            freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1/self.sample_rate)).astype(np.float32)
        self.freqs = freqs
        self.magnitude = self.fft()

    def fft(self):
        fft_result = np.fft.fftshift(np.fft.fft(self.samples, n=self.fft_size))
        magnitude = 20 * np.log10(np.abs(fft_result) + 1e-12)
        return magnitude

    def apply_gaussian_filter(self, sigma):
        self.magnitude = gaussian_filter1d(self.magnitude, sigma=sigma)

    def apply_median_filter(self, kernel_size):
        self.magnitude = medfilt(self.magnitude, kernel_size=kernel_size)

    def apply_smooth_moving_average(self, window_size):
        kernel = np.ones(window_size) / window_size
        self.magnitude = np.convolve(self.magnitude, kernel, mode='same').astype(np.float32)

    def apply_exponential_moving_average(self, previous_magnitude, alpha=0.3):
            if previous_magnitude is None:
                return self.magnitude
            return alpha * self.magnitude + (1 - alpha) * previous_magnitude
