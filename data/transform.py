import numpy as np
from scipy.signal import resample
from pycbc.types import TimeSeries
from pycbc.psd import interpolate, inverse_spectrum_truncation, from_string
from pycbc.psd import aLIGOZeroDetHighPower as aPSD
from pycbc.filter import sigma
from BnsLib.utils.formatting import input_to_list, list_length

def multi_rate_sample(ts, samples_per_part, sample_rates, reverse=False, keep_end=True):
    """Function to re-sample a given time series at multiple rates.
    
    Arguments
    ---------
    ts : pycbc.TimeSeries
        The time series to be re-sampled.
    samples_per_part : int
        How many samples each part should contain.
    sample_rates : list of int
        A list of the sample rates that should be used.
        If reverse is False the first entry corresponds
        to the sample rate that should be used for the
        final part of the time series. The second entry
        corresponds to the part prior to the final part
        and does not overlap.
    reverse : {bool, False}
        Set this to True, if the first sample rate in
        sample_rates should re-sample the inital part of
        the time series. (i.e. re-sampling happens in
        a time-ordered manner)
    keep_end : {bool, True}
        If the re-sampled time series is shorter than the
        original time series, this option specifies if
        the original time-series is cropped in the beginning
        or end. (Default: cropped in the beginning)
    
    Returns
    -------
    re-sampled : list of pycbc.TimeSeries
        A list of pycbc.TimeSeries containing the re-sampled
        data. The time series are ordered, such that the
        first list entry corresponds to the initial part
        of the time series and the final entry corresponds
        to the final part of the waveform.
    
    Examples
    --------
    -We want to re-sample a time series of duration 10s.
     Each part should 400 samples. We want to sample
     the initial part of the time series with a sample
     rate of 50 and the part after that using a sample
     rate of 200. Therefore second 0 to 8 would be sampled
     at a rate of 50 and second 8 to 10 using a rate of 200.
     We could use the call:
     multi_rate_sample(ts, 400, [200, 50])
     or
     multi_rate_sample(ts, 400, [50, 200], reverse=True)
     
     We would receive the return
     [TimeSeries(start_time=0, end_time=8, sample_rate=50),
      TimeSeries(start_time=8, end_time=10, sample_rate=200)]
     in both cases.
    
    -We want to re-sample a time series of duration 10s. We want
     each part to contain 400 samples and want to use the sample
     rates [400, 50]. The re-sampled time series would be of
     total duration 9s, as sampling 400 samples with a rate of
     400 yields 1s and sampling 400 samples with a rate of 50 would
     yield 8s. The function call would be wither
     multi_rate_sample(ts, 400, [400, 50])
     or
     multi_rate_sample(ts, 400, [50, 400], reverse=True)
     
     with the output
     [TimeSeries(start_time=1, end_time=9, sample_rate=50),
      TimeSeries(start_time=9, end_time=10, sample_rate=200)]
    """
    if reverse:
        sample_rates = sample_rates.copy()
        sample_rates.reverse()
    sample_rates = np.array(sample_rates, dtype=int)
    samples_per_part = int(samples_per_part)
    durations = float(samples_per_part) / sample_rates
    total_duration = sum(durations)
    if total_duration > ts.duration:
        msg = 'Cannot re-sample a time series of duration '
        msg += f'{ts.duration} with sample-rates {sample_rates} and '
        msg += f'samples per part {sample_per_part}.'
        ValueError(msg)
    parts = []
    last_time = ts.end_time
    if not keep_end:
        sample_rates = list(sample_rates)
        sample_rates.reverse()
        sample_rates = np.array(sample_rates, dtype=int)
        last_time = ts.start_time
    for i, sr in enumerate(sample_rates):
        resampled_data, resampled_t = resample(ts.data,
                                               int(len(ts) / ts.sample_rate * sr),
                                               t=ts.sample_times)
        delta_t = resampled_t[1] - resampled_t[0]
        epoch = resampled_t[0]
        resampled = TimeSeries(resampled_data, delta_t=delta_t,
                               epoch=epoch)
        if keep_end:
            diff = samples_per_part / sr
            st = max(last_time-2*diff, ts.start_time)
            tmp = resampled.time_slice(st, last_time)
            sidx = len(tmp) - samples_per_part
            eidx = len(tmp)
            parts.append(tmp[sidx:eidx])
            last_time = parts[-1].start_time
        else:
            diff = samples_per_part / sr
            et = min(last_time+2*diff, ts.end_time)
            tmp = resampled.time_slice(last_time, et)
            sidx = 0
            eidx = samples_per_part
            parts.append(tmp[sidx:eidx])
            last_time = parts[-1].end_time
    if keep_end:
        parts.reverse()
    return parts

def whiten(strain_list, low_freq_cutoff=20., max_filter_duration=4.,
           psd=None):
    """Returns the data whitened by the PSD.
    
    Arguments
    ---------
    strain_list : pycbc.TimeSeries or list of pycbc.TimeSeries
        The data that should be whitened.
    low_freq_cutoff : {float, 20.}
        The lowest frequency that is considered during calculations. It
        must be >= than the lowest frequency where the PSD is not zero.
        Unit: hertz
    max_filter_duration : {float, 4.}
        The duration to which the PSD is truncated to in the
        time-domain. The amount of time is removed from both the
        beginning and end of the input data to avoid wrap-around errors.
        Unit: seconds
    psd : {None or str or pycbc.FrequencySeries, None}
        The PSD that should be used to whiten the data. If set to None
        the pycbc.psd.aLIGOZeroDetHighPower PSD will be used. If a PSD
        is provided which does not fit the delta_f of the data, it will
        be interpolated to fit. If a string is provided, it will be
        assumed to be known to PyCBC.
    
    Returns
    -------
    pycbc.TimeSeries or list of pycbc.TimeSeries
        Depending on the input type it will return a list of TimeSeries
        or a single TimeSeries. The data contained in this time series
        is the whitened input data, where the inital and final seconds
        as specified by max_filter_duration are removed.
    """
    org_type = type(strain_list)
    if not org_type == list:
        strain_list = [strain_list]
    
    ret = []
    for strain in strain_list:
        df = strain.delta_f
        f_len = int(len(strain) / 2) + 1
        if psd is None:
            psd = aPSD(length=f_len,
                       delta_f=df,
                       low_freq_cutoff=low_freq_cutoff-2.)
        elif isinstance(psd, str):
            psd = from_string(psd,
                              length=f_len,
                              delta_f=df,
                              low_freq_cutoff=low_freq_cutoff-2.)
        else:
            if not len(psd) == f_len:
                msg = 'Length of PSD does not match data.'
                raise ValueError(msg)
            elif not psd.delta_f == df:
                psd = interpolate(psd, df)
        max_filter_len = int(max_filter_duration * strain.sample_rate) #Cut out the beginning and end
        psd = inverse_spectrum_truncation(psd,
                                          max_filter_len=max_filter_len,
                                          low_frequency_cutoff=low_freq_cutoff,
                                          trunc_method='hann')
        f_strain = strain.to_frequencyseries()
        kmin = int(low_freq_cutoff / df)
        f_strain.data[:kmin] = 0
        f_strain.data[-1] = 0
        f_strain.data[kmin:] /= psd[kmin:] ** 0.5
        strain = f_strain.to_timeseries()
        ret.append(strain[max_filter_len:len(strain)-max_filter_len])
    
    if not org_type == list:
        return(ret[0])
    else:
        return(ret)

def rescale_snr(signal, old_snr, new_snr):
    """Rescale a pycbc.TimeSeries or pycbc.FrequencySeries to a given
    signal-to-noise ratio.
    
    Arguments
    ---------
    signal : pycbc.TimeSeries or pycbc.FrequencySeries
        The data that should be rescaled.
    old_snr : float
        The signal-to-noise ratio of the input signal.
    new_snr : float
        The signal-to-noise ratio the output signal should have.
    
    Returns
    -------
    pycbc.TimeSeries or pycbcFrequencySeries
        Returns the signal (same data-type as input) after rescaling.
    """
    return signal / old_snr * new_snr

def optimal_snr(signal, psd='aLIGOZeroDetHighPower',
                low_freq_cutoff=None, high_freq_cutoff=None):
    """Calculate the optimal signal-to-noise ratio for a given signal.
    
    Arguments
    ---------
    signal : pycbc.TimeSeries or pycbc.FrequencySeries
        The signal of which to calculate the signal-to-noise ratio.
    psd : {str or None or pycbc.FrequencySeries, 'aLIGOZeroDetHighPower}
        A power spectral density to use for the noise-model. If set to a
        string, a power spectrum will be generated using
        pycbc.psd.from_string. If set to None, no noise will be assumed.
        If a frequency series is given, the user has to make sure that
        the delta_f and length match the signal.
    low_freq_cutoff : {float or None, None}
        The lowest frequency to consider. If a value is given, the power
        spectrum will be generated with a lower frequency cutoff 2 below
        the given one. (0 at minimum)
    high_freq_cutoff : {float or None, None}
        The highest frequency to consider.
    
    Returns
    -------
    float
        The optimal signal-to-noise ratio given the signal and the noise
        curve (power spectrum).
    """
    if psd is not None:
        if isinstance(psd, str):
            df = signal.delta_f
            if isinstance(signal, TimeSeries):
                flen = len(signal) // 2 + 1
            elif isinstance(signal, FrequencySeries):
                flen = len(signal)
            psd_low = 0. if low_freq_cutoff is None else max(low_freq_cutoff - 2., 0.)
            psd = from_string(psd, length=flen, delta_f=df,
                              low_freq_cutoff=psd_low)
    return sigma(signal, psd=psd, low_frequency_cutoff=low_freq_cutoff,
                 high_frequency_cutoff=high_freq_cutoff)

def optimal_network_snr(signals, psds='aLIGOZeroDetHighPower',
                        low_freq_cutoffs=None, high_freq_cutoffs=None):
    """Returns the optimal network signal-to-noise ratio for a list
    of signals and noise curves.
    
    signals : list of (pycbc.TimeSeries or pycbc.FrequencySeries)
        A list of the signals to analyze. This argument may be a single
        time or frequency series.
    psds : {list of (str or None or pycbc.FrequencySeries), 'aLIGOZeroDetHighPower'}
        A list of power spectrums to use for the different signals. The
        length has to match the length of signals. May be a single value
        instead of a list. If a single value is provided, the same value
        is assumed for all signals. All entries have to be interpretable
        by the function optimal_snr.
    low_freq_cutoffs : {list of (float or None), None}
        A list of low frequency cutoffs. See the documentation of
        optimal_snr for more details on this option. May be a single
        value. If a single value is provided, the same value is assumed
        for all signals.
    high_freq_cutoffs : {list of (float or None), None}
        A list of high frequency cutoffs. See the documentation of
        optimal_snr for more details on this option. May be a single
        value. If a single value is provided, the same value is assumed
        for all signals.
    
    Returns
    -------
    float
        The optimal network signal-to-noise ratio given the signals and
        the noise curves (power spectra).
    """
    length = list_length(signals)
    signals = input_to_list(signals, length=length)
    psds = input_to_list(psds, length=length)
    low_freq_cutoffs = input_to_list(low_freq_cutoffs, length=length)
    high_freq_cutoffs = input_to_list(high_freq_cutoffs, length=length)
    snrs = []
    for i in range(length):
        snrs.append(optimal_snr(signals[i],
                                psd=psds[i],
                                low_freq_cutoff=low_freq_cutoffs[i],
                                high_freq_cutoff=high_freq_cutoffs[i]))
    return np.sqrt(np.sum(np.square(snrs)))
