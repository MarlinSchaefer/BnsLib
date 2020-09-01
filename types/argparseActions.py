import argparse
#########
#Actions#
#########

class TranslationAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        tmp = {}
        for pt in values:
            key, val = pt.split(':')
            tmp[key] = val
        setattr(namespace, self.dest, tmp)

#######
#Types#
#######

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
