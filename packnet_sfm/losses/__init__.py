"""Loss modules exported by :mod:`packnet_sfm.losses`."""

from .scale_adaptive_loss import ScaleAdaptiveLoss
from .adaptive_multi_domain_loss import AdaptiveMultiDomainLoss, AdaptiveMultiDomainLossWrapper
from .fixed_multi_domain_loss import FixedMultiDomainLoss, FixedMultiDomainLossWrapper

__all__ = [
	'ScaleAdaptiveLoss',
	'AdaptiveMultiDomainLoss',
	'AdaptiveMultiDomainLossWrapper',
	'FixedMultiDomainLoss',
	'FixedMultiDomainLossWrapper',
]
