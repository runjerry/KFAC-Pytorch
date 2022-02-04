from .kfac import KFACOptimizer
from .ekfac import EKFACOptimizer
from .ker_kfac import KerKFACOptimizer


def get_optimizer(name):
    if name == 'kfac':
        return KFACOptimizer
    elif name == 'ker_kfac':
        return KerKFACOptimizer
    elif name == 'ekfac':
        return EKFACOptimizer
    else:
        raise NotImplementedError
