import numpy as np
from pycbc.waveform import get_td_waveform, get_fd_waveform
from BnsLib.utils.formatting import input_to_list, list_length
from functools import wraps
from pycbc.detector import Detector
import warnings

class WaveformGetter(object):
    def __init__(self, variable_params=None, static_params=None,
                 num_processes=1, domain='time', detectors='H1'):
        self.initialized = False
        self._num_processes = num_processes
        self.variable_params = variable_params
        self.static_params = static_params
        self.domain = domain
        self.detector_names = detectors
        #if self.detectors is None:
            #self.detector_names = None
        #else:
            #self.detector_names = [det.name for det in self.detectors]
        self._it_index = 0
    
    def __len__(self):
        if len(self.variable_params) == 0:
            if len(self.static_params) == 0:
                return 0
            else:
                return 1
        else:
            key = list(self.variable_params.keys())[0]
            return len(self.variable_params[key])
    
    def __getitem__(self, index):
        return self.generate(index=index)
    
    def __next__(self):
        if self._it_index < len(self):
            ret = self[self._it_index]
            self._it_index += 1
            return ret
        else:
            raise StopIteration
    
    def __iter__(self):
        return self
    
    def generate(self, index=None, single_detector_as_list=True):
        """Generates one or multiple waveforms.
        
        Arguments
        ---------
        index : {int or slice or None, None}
            Which waveforms to generate. If set to None, all waveforms
            will be generated. Indices point to the given lists
            variable_params.
        single_detector_as_list : {bool, True}
            Usually this function will return a dictionary of lists,
            where each entry corresponds to one of multiple detectors.
            If only a single detector is used it is not necessary to
            use a dictionary. If this option is set to true, only the
            value of the dictionary will be returned when a single
            detector is used.
        
        Returns
        -------
        dict of list or list or pycbc.TimeSeries:
            The return type depends on the index and the option
            `single_detector_as_list`. If multiple detectors are used
            and the index is a slice, a dictionary of lists will be
            returned. The keys to the dictionary contain the detector
            prefixes and the lists contain transformed waveforms [1].
            If the index is an integer instead the values of the
            dictionary will not be lists but the transformed waveform
            instead. If the option `single_detector_as_list` is set to
            True and only a single detector is provided the function
            will return just the waveform and no dictionary.
        """
        was_int = False
        if index is None:
            index = slice(None, None)
        ret_idx = index
        if isinstance(index, int):
            index = slice(index, index+1)
            was_int = True
        
        if self.detector_names is None:
            keys = ['plus', 'cross']
        else:
            keys = self.detector_names
            
            ra_key = None
            if 'ra' in self.variable_params or 'ra' in self.static_params:
                ra_key = 'ra'
            elif 'right_ascension' in self.variable_params or 'right_ascension' in self.static_params:
                ra_key = 'right_ascension'
            if ra_key is None:
                ra_key = 'right_ascension'
                self.variable_params[ra_key] = [np.nan for _ in range(len(self))]
                msg = 'right_ascension was set neither in '
                msg += 'variable_params nor in static_params. When '
                msg += 'detectors are specified right_ascension should '
                msg += 'be set manually. Will use the optimal source '
                msg += 'right_ascensions instead.'
                warnings.warn(msg, RuntimeWarning)
            
            dec_key = None
            if 'dec' in self.variable_params or 'dec' in self.static_params:
                dec_key = 'dec'
            elif 'declination' in self.variable_params or 'declination' in self.static_params:
                dec_key = 'declination'
            if dec_key is None:
                dec_key = 'declination'
                self.variable_params[dec_key] = [np.nan for _ in range(len(self))]
                msg = 'declination was set neither in '
                msg += 'variable_params nor in static_params. When '
                msg += 'detectors are specified declination should '
                msg += 'be set manually. Will use the optimal source '
                msg += 'declinations instead.'
                warnings.warn(msg, RuntimeWarning)
            
            pol_key = None
            if 'pol' in self.variable_params or 'pol' in self.static_params:
                pol_key = 'pol'
            elif 'polarization' in self.variable_params or 'polarization' in self.static_params:
                pol_key = 'polarization'
            if pol_key is None:
                pol_key = 'polarization'
                self.static_params[pol_key] = 0.
                msg = 'polarization was set neither in '
                msg += 'variable_params nor in static_params. When '
                msg += 'detectors are specified declination should '
                msg += 'be set manually. Will use polarization 0 '
                msg += 'instead.'
                warnings.warn(msg, RuntimeWarning)
        
        ret = {key: [] for key in keys}
        for i in range(*index.indices(len(self))):
            wave_kwargs = {}
            for key, val in self.static_params.items():
                wave_kwargs[key] = val
            for key, val in self.variable_params.items():
                wave_kwargs[key] = val[i]
            if self.domain == 'time':
                hp, hc = get_td_waveform(**wave_kwargs)
            elif self.domain == 'frequency':
                hp, hc = get_fd_waveform(**wave_kwargs)
            if self.detector_names is None:
                ret['plus'].append(self.transform(hp))
                ret['cross'].append(self.transform(hc))
            else:
                if ra_key not in self.static_params:
                    if len(self.variable_params[ra_key]) <= i:
                        for _ in range(i+1-len(self.variable_params[ra_key])):
                            self.variable_params[ra_key].append(np.nan)
                if dec_key not in self.static_params:
                    if len(self.variable_params[dec_key]) <= i:
                        for _ in range(i+1-len(self.variable_params[dec_key])):
                            self.variable_params[dec_key].append(np.nan)
                for det_name in self.detector_names:
                    det = Detector(det_name)
                    st = float(hp.start_time)
                    calc_opt = {ra_key: np.isnan(wave_kwargs[ra_key]),
                                dec_key: np.isnan(wave_kwargs[dec_key])}
                    if any(list(calc_opt.values())):
                        opt_ra, opt_dec = det.optimal_orientation(st)
                        if calc_opt[ra_key]:
                            wave_kwargs[ra_key] = opt_ra
                            if ra_key in self.variable_params:
                                self.variable_params[ra_key][i] = opt_ra
                        if calc_opt[dec_key]:
                            wave_kwargs[dec_key] = opt_dec
                            if dec_key in self.variable_params:
                                self.variable_params[dec_key][i] = opt_dec
                    fp, fc = det.antenna_pattern(wave_kwargs[ra_key],
                                                 wave_kwargs[dec_key],
                                                 wave_kwargs[pol_key],
                                                 st)
                    h = fp * hp + fc * hc
                    ret[det.name].append(self.transform(h))
        
        if was_int:
            ret = {key: val[0] for (key, val) in ret.items()}
        
        if self.detector_names is None:
            return (ret_idx, ret)
        
        if single_detector_as_list and len(self.detector_names) == 1:
            return (ret_idx, ret[self.detector_names[0]])
        return (ret_idx, ret)
    
    def get_params(self, index=None):
        if index is None:
            index = slice(None, None)
        ret = {}
        if isinstance(index, int):
            for key, val in self.static_params.items():
                ret[key] = val
            for key, val in self.variable_params.items():
                ret[key] = val[index]
        elif isinstance(index, slice):
            slice_size = len(range(len(self))[index])
            for key, val in self.static_params.items():
                ret[key] = [val for _ in range(slice_size)]
            for key, val in self.variable_params.items():
                ret[key] = val[index]
        return ret
    
    def transform(self, wav):
        return wav
    
    def assert_not_initialized(self, orig_func):
        @wraps(orig_func)
        def wrapper(*args, **kwargs):
            if self.initialized:
                return None
            else:
                return orig_func(*args, **kwargs)
        return wrapper
    
    @property
    def num_processes(self):
        return self._num_processes
    
    @num_processes.setter
    def num_processes(self, num_processes):
        if num_processes is None:
            num_processes = 1
        
        if not isinstance(num_processes, int):
            msg = 'num_processes must be of type int. Got '
            msg += f'{type(num_processes)} instead.'
            raise TypeError(msg)
        
        if num_processes < 1:
            msg = f'num_processes must be greater 0. Got {num_processes}.'
            raise ValueError(msg)
        
        self._num_processes = num_processes
    
    @property
    def variable_params(self):
        return self._variable_params
    
    @variable_params.setter
    def variable_params(self, variable_params):
        if variable_params is None:
            self._variable_params = {}
        if not isinstance(variable_params, dict):
            msg = 'variable_params must be a dictionary containing '
            msg += 'iterables of the same length. Got type '
            msg += f'{type(variable_params)} instead.'
            raise TypeError(msg)
        parts = list(variable_params.values())
        if not all([len(pt) == len(parts[0]) for pt in parts]):
            msg = 'variable_params must be a dictionary containing '
            msg += 'iterables of the same length. Got lengths '
            msg_dict = {key: len(val) for (key, val) in variable_params.items()}
            msg += f'{msg_dict}.'
            raise ValueError(msg)
        self._variable_params = variable_params
    
    @property
    def static_params(self):
        return self._static_params
    
    @static_params.setter
    def static_params(self, static_params):
        if static_params is None:
            self._static_params = {}
        if not isinstance(static_params, dict):
            msg = 'static_params must be a dictionary. Got type '
            msg += f'{type(static_params)} instead.'
            raise TypeError(msg)
        self._static_params = static_params
    
    @property
    def domain(self):
        return self._domain
    
    @domain.setter
    def domain(self, domain):
        time_domains = ['time', 't']
        freq_domains = ['frequency', 'freq', 'f']
        poss_domains = time_domains + freq_domains
        if domain.lower() not in poss_domains:
            msg = f'domain must be one of {poss_domains}, not {domain}.'
            raise ValueError(msg)
        if domain.lower() in time_domains:
            self._domain = 'time'
        
        if domain.lower()in freq_domains:
            self._domain = 'frequency'
    
    @classmethod
    def from_config(cls, config_file, number_samples):
        return

from pycbc.workflow import WorkflowConfigParser
from pycbc.distributions import read_params_from_config
from pycbc.distributions import read_distributions_from_config
from pycbc.distributions import read_constraints_from_config
from pycbc.distributions import JointDistribution
from pycbc.transforms import read_transforms_from_config, apply_transforms
class WFParamGenerator(object):
    def __init__(self, config_file, seed=0):
        np.random.seed(seed)
        config_file = input_to_list(config_file)
        config_file = WorkflowConfigParser(config_file, None)
        var_args, self.static = read_params_from_config(config_file)
        constraints = read_constraints_from_config(config_file)
        dist = read_distributions_from_config(config_file)

        self.trans = read_transforms_from_config(config_file)
        self.pval = JointDistribution(var_args, *dist, 
                                **{"constraints": constraints})   

    def draw(self):
        return apply_transforms(self.pval.rvs(), self.trans)
    
    def draw_full(self):
        var_draws = apply_transforms(self.pval.rvs(), self.trans)
        ret = self.static.copy()
        ret.update(dict(var_draws))
        return ret

class WaveformGenerator(WaveformGetter):
    def __init__(self, config_file, seed=0, domain='time',
                 detectors='H1'):
        self.params = WFParamGenerator(config_file, seed=seed)
        vp = {key: [] for key in self.params.pval.variable_args}
        sp = self.params.static
        super().__init__(variable_params=vp,
                         static_params=sp,
                         domain=domain,
                         detectors=detectors)
    
    def __next__(self):
        params = self.params.draw()
        for key in self.variable_params.keys():
            if key in params:
                self.variable_params[key].append(params[key][0])
        return self[len(self)-1]
