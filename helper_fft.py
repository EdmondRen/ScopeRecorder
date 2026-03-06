

import numpy as np
import scipy as sp

def make_logbin_map(n_fft, fs, n_log_bins=512, fmin=0.0, fmax=None, K=2):
    """
    Precompute mapping from linear FFT bins -> log-spaced bins.
    Returns:
      bin_id: int array of length n_freq (rfft bins) with log-bin index (or -1)
      counts: float array length n_log_bins (# of linear bins per log bin)
      f_centers: float array length n_log_bins (Hz, geometric centers)
    """
    n_freq = n_fft // 2 + 1
    freqs = np.fft.rfftfreq(n_fft, d=1.0/fs)

    if fmax is None:
        fmax = fs / 2.0
    fmin = max(fmin, freqs[1] if n_freq > 1 else fmin)

    # Log-spaced edges (Hz)
    edges = np.geomspace(fmin, fmax, num=n_log_bins + 1)
    bin_size = np.diff(edges)
    f_centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean center
    switch_frequency_ind = np.argmax(bin_size>(freqs[1]-freqs[0]) * K)

    # Assign each linear bin to a log bin
    bin_id = np.searchsorted(edges, freqs, side="right") - 1
    valid = (bin_id >= 0) & (bin_id < n_log_bins)
    bin_id = np.where(valid, bin_id, -1)

    counts = np.bincount(bin_id[valid], minlength=n_log_bins).astype(np.float32)
    counts[counts == 0] = 1.0  # avoid divide-by-zero

    return bin_id.astype(np.int32), counts, f_centers, switch_frequency_ind

def rebin_to_log(power_lin, bin_id, counts, f_centers, switch_frequency_ind):
    """
    power_lin: 1D array length n_freq (linear power for rfft bins)
    returns: 1D array length n_log_bins (mean power per log bin)
    """
    valid = bin_id >= 0
    summed = np.bincount(bin_id[valid], weights=power_lin[valid], minlength=len(counts)).astype(np.float32)
    summed = summed / counts
    
    # Linear region
    freqs_lin = np.fft.rfftfreq((len(power_lin)-1)*2, d=1.0)
    idx = np.argmax(freqs_lin>f_centers[switch_frequency_ind])

    f_interp = sp.interpolate.interp1d(freqs_lin[:idx], power_lin[:idx])
    for i in range(switch_frequency_ind):
        # summed[i] = power_lin[np.argmax(freqs_lin>f_centers[i])]
        summed[i] = f_interp(f_centers[i])

    return summed


def fft_logbin(pxx, bins=16384, fs = 1):
    bin_id, counts, f_centers, switch_frequency_ind = make_logbin_map((len(pxx)-1)*2, 1, n_log_bins=bins, fmin=0, fmax=1/2)
    pxx_logx = rebin_to_log(pxx, bin_id, counts, f_centers, switch_frequency_ind)
    return f_centers * fs, pxx_logx



class LogSpectrogram:
    def __init__(self, trace_len, bins = 2048, fs = 1, window=None):
        self.fs = fs
        self.f_centers_lin = np.fft.rfftfreq(trace_len, d = 1/fs)
        self.bin_id, self.counts, self.f_centers_log, self.switch_frequency_ind = make_logbin_map(trace_len, 1, n_log_bins=bins, fmin=self.f_centers_lin[1], fmax=self.f_centers_lin[-1])        
        self.data = []
        
    def add(self, fft_power_lin):
        fft_log_power = rebin_to_log(fft_power_lin, self.bin_id, self.counts, self.f_centers_log, self.switch_frequency_ind)
        self.data.append(fft_log_power)

    def get_data(self):
        return self.f_centers_log, self.data