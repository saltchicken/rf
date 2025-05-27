import numpy as np
from scipy.signal import firwin, lfilter, medfilt, resample_poly
from scipy.ndimage import gaussian_filter1d
from scipy.io.wavfile import write

class Sample:
    def __init__(self, samples, sample_rate):
        self.samples = samples
        self.sample_rate = sample_rate

    def apply_resample(self, down_sample_rate):
        decimation_factor = int(self.sample_rate / down_sample_rate)
        self.samples = resample_poly(self.samples, up=1, down=decimation_factor)
        self.sample_rate = down_sample_rate

class Signal(Sample):
    def __init__(self, samples, sample_rate):
        super().__init__(samples, sample_rate)

    def apply_freq_offset(self, freq_offset):
        t = np.arange(len(self.samples)) / self.sample_rate
        self.samples = self.samples * np.exp(-1j * 2 * np.pi * freq_offset * t)

    def apply_low_pass_filter(self, cutoff_hz):
        nyquist_rate = self.sample_rate / 2
        num_taps = 101
        fir_coeff = firwin(num_taps, cutoff_hz / nyquist_rate)
        self.samples = lfilter(fir_coeff, 1.0, self.samples)

class FFT(Sample):
    def __init__(self, samples, sample_rate, freqs=None):
        super().__init__(samples, sample_rate)
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

class FM(Sample):
    def __init__(self, samples, sample_rate):
        super().__init__(samples, sample_rate)

    # def apply_fm_demodulation(self, frequency_deviation):
    #     t = np.arange(len(self.samples)) / self.sample_rate
    #     fm_signal = np.cos(2 * np.pi * frequency_deviation * t)
    #     demodulated_signal = self.samples * fm_signal
    #     self.samples = demodulated_signal
    # def fm_demodulate(iq):
    #     angle = np.angle(iq[1:] * np.conj(iq[:-1]))
    #     return angle
    def apply_fm_demodulation(self):
        self.samples = np.angle(self.samples[1:] * np.conj(self.samples[:-1]))

    def apply_normalization(self):
        self.samples /= np.max(np.abs(self.samples))

    def convert_to_int16(self):
        self.samples = np.int16(self.samples * 32767)

    def apply_volume(self, volume):
        self.samples *= volume

    def save(self, output_filename):
        write(output_filename, int(self.sample_rate), self.samples)
        print(f"Signal saved to {output_filename} - Sample Rate: {self.sample_rate} Hz")
