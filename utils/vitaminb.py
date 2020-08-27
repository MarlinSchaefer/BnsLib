import h5py
from vitaminb.params_files import make_params_files
import warnings

massTrans = {'mass1': 'mass_1',
             'mass2': 'mass_2',
             'mass_1': 'mass_1',
             'mass_2': 'mass_2'}

pycbc_to_vitamin = {'mass1': 'mass_1',
                    'mass2': 'mass_2',
                    'distance': 'luminosity_distance',
                    'ra': 'ra',
                    'dec': 'dec',
                    'tc': 'geocent_time',
                    'inclination': 'theta_jn',
                    'coa_phase': 'phase',
                    'pol': 'psi',
                    'mass_1': 'mass_1',
                    'mass_2': 'mass_2'}

vitamin_to_pycbc = {pycbc_to_vitamin[key]: key for key in pycbc_to_vitamin.keys()}

def params_files_from_config(config_file, translation):
    from BnsLib.data.genenerate_train import WFParamGenerator
    #-Save all parameters to a dict {name: bounds}
    #-Go through the list of transformations and apply:
    #    BnsLib.utils.bounds.estimate_transformed_bounds
    # to them
    #-Go through transformations and parameters and look which are
    # actually output -> save into the params dictionary
    #-Apply translation to params, bounds and fixed-params
    params = make_params_files.get_params()
    bounds = {}
    fixed_vals = {}
    gen = WFParamGenerator(config_file)
    for dist in gen.params.pval.distributions:
        for key, val in dist.items():
            if 'mass' in key:
            key = massTrans[key]
        bounds[key + '_min'] = val.min
        bounds[key + '_max'] = val.max
        bounds['__definition__' + key] = f'{key} range'
        fixed_vals['__definition__' + key] = f'{key} fixed value'
        fixed_vals[key] = (val.max + val.min) / 2
    
