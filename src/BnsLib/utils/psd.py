import os
import numpy as np
from pycbc.types import FrequencySeries
from pycbc.psd import from_string, from_numpy_arrays, interpolate
import h5py


def apply_low_freq_cutoff(freqseries, low_freq_cutoff):
    if low_freq_cutoff is None:
        return freqseries
    idx = np.searchsorted(freqseries.sample_frequencies, low_freq_cutoff)
    freqseries[:idx] = 0
    return freqseries


def apply_delta_f(freqseries, delta_f):
    if delta_f is None:
        return freqseries
    return interpolate(freqseries, delta_f)


def load_psd_file(path, flen=None, delta_f=None, low_freq_cutoff=None,
                  is_asd_file=False):
    try:
        with h5py.File(path, 'r') as f:
            data = f['data'][:]
            delta_f = f['data'].attrs['delta_f']
            if 'epoch' in f['data'].attrs:    
                epoch = f['data'].attrs['epoch']
            else:
                epoch = None
            psd = FrequencySeries(data, delta_f=delta_f, epoch=epoch)
        if is_asd_file:
            psd = psd ** 2
        if flen is None:
            psd = apply_delta_f(psd, delta_f)
            psd = apply_low_freq_cutoff(psd, low_freq_cutoff)
        else:
            if delta_f is None:
                delta_f = psd.delta_f
            if low_freq_cutoff is None:
                low_freq_cutoff = 0.       
            psd = from_numpy_arrays(psd.sample_frequencies,
                                    psd.numpy(),
                                    flen,
                                    delta_f,
                                    low_freq_cutoff)
    except (OSError, ValueError):
        file_data = np.loadtxt(path)
        if (file_data < 0).any() or \
           np.logical_not(np.isfinite(file_data)).any():
            raise ValueError('Invalid data in ' + path)

        freq_data = file_data[:, 0]
        noise_data = file_data[:, 1]
        
        if is_asd_file:
            noise_data = noise_data ** 2
        if delta_f is None:
            # delta_f = (freq_data.max() - freq_data.min()) / len(freq_data)
            delta_f = (freq_data[1:] - freq_data[:-1]).min()
        if flen is None:
            flen = int(freq_data.max() // delta_f)
        if low_freq_cutoff is None:
            low_freq_cutoff = max(0., freq_data.min()+10*delta_f)
        psd = from_numpy_arrays(freq_data, noise_data, flen, delta_f,
                                low_freq_cutoff)
    return psd


def get_psd(psd, flen=None, delta_f=None, low_freq_cutoff=None,
            is_asd=False):
    """Return a PSD with the given parameters.
    
    Arguments
    ---------
    psd : str or FrequencySeries
        If a str, it has to be either a file-path from which to load the
        PSD or a known name of a PSD.
    flen : {None or int, None}
        Number of samples in the PSD. If None it has to be loaded from
        a file and the number of samples is determined from that.
    delta_f : {None or float, None}
        The delta_f of the output PSD. If None it has to be loaded from
        a file and the delta_f is determined from that.
    low_freq_cutoff : {None or float, None}
        The low frequency cutoff to apply to the PSD. If None it will be
        set to 0.
    is_asd : {bool, False}
        Whether or not the data is an ASD.
    """
    if isinstance(psd, str):
        if os.path.isfile(psd):
            try:
                psd = load_psd_file(psd, flen=flen, delta_f=delta_f,
                                    low_freq_cutoff=low_freq_cutoff,
                                    is_asd_file=is_asd)
            except (ValueError, OSError):
                if flen is None or delta_f is None or \
                   low_freq_cutoff is None:
                    msg = 'Must provide flen, delta_f and low_freq_cutoff.'
                    raise ValueError(msg)
                psd = from_string(psd, flen, delta_f, low_freq_cutoff)
        else:
            if flen is None or delta_f is None or \
               low_freq_cutoff is None:
                msg = 'Must provide flen, delta_f and low_freq_cutoff.'
                raise ValueError(msg)
            psd = from_string(psd, flen, delta_f, low_freq_cutoff)
    elif isinstance(psd, FrequencySeries):
        if is_asd:
            psd = psd ** 2
        if flen is not None:
            if delta_f is None:
                delta_f = psd.delta_f
            if low_freq_cutoff is None:
                low_freq_cutoff = 0.
            psd = from_numpy_arrays(psd.sample_frequencies,
                                    psd.numpy(),
                                    flen,
                                    delta_f,
                                    low_freq_cutoff)
        else:
            psd = apply_delta_f(psd, delta_f)
            psd = apply_low_freq_cutoff(psd, low_freq_cutoff)
    return psd
