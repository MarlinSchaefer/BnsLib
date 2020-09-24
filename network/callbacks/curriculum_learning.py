from tensorflow import keras
from BnsLib.utils.math import safe_min, safe_max

class SnrCurriculumLearningScheduler(keras.callbacks.Callback):
    """A callback that schedules when the SNR of the data-samples should
    be re-scaled when training a neural network.
    
    Arguments
    ---------
    generator : keras.utils.Sequence or Generator
        The generator that provides the data-samples to the network.
        This generator needs to have an attribute `target` and a method
        `rescale`. For details see the notes section.
    lower_at : {int or float, 10}
        Specifies when the SNR should be lowered. If set to an integer
        the SNR will be lowered after that number of epochs regardless
        of the loss. When set to a float the SNR will be lowered once
        the total validation loss drops below this value.
    lower_by : {float, 5.}
        The value by which the SNR should be lowered.
    min_snr : {float, 5.}
        The SNR-value at which the training SNR will not be lowered
        anymore.
    
    Notes
    -----
    -The `target` attribute of the generator should be a list of length
     2, where the first entry gives the lower bound and the second entry
     gives the upper bound. The SNR will be drawn from this range. To
     use a single value simply set the upper bound equal to the lower
     bound.
    -The `rescale` method must take a list of length 2 as argument. The
     list contains the new target value.
    """
    def __init__(self, generator, lower_at=10, lower_by=5., min_snr=5.):
        self.generator = generator
        self.lower_at = lower_at
        self.lower_by = lower_by
        self.min_snr = min_snr
    
    def on_epoch_begin(self, epoch, logs=None):
        print("Training with target: {}".format(self.generator.target))
    
    def on_epoch_end(self, epoch, logs={}):
        lower = False
        if isinstance(self.lower_at, int):
            lower = (epoch > 0) and (epoch % self.lower_at == 0)
        elif isinstance(self.lower_at, float):
            loss = logs.get('val_loss', 0.)
            lower = (loss < self.lower_at)
        if lower:
            target = self.generator.target
            minsnr = safe_min(target)
            maxsnr = safe_max(target)
            newmin = max(self.min_snr, minsnr - self.lower_by)
            newmax = newmin + maxsnr - minsnr
            if newmin == newmax:
                new_target = newmin
            else:
                new_target = [newmin, newmax]
            self.generator.rescale(new_target)
            print('\nSet SNR of generator {} to {} on epoch {}.'.format(self.generator.__class__.__name__, new_target, epoch))
