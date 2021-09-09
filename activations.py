import torch
from torch.nn.functional import softmax
from functools import partial
from entmax import sparsemax, entmax15, entmax_bisect


def alpha_entmax(alpha):
	"""
	Returns the alpha-entmax transformation for alpha >= 1.
	Particular cases:
		- alpha = 1 gives torch.nn.functional.softmax
		- alpha = 1.5 gives entmax.entmax15
		- alpha = 2 gives entmax.sparsemax
	For other alpha this is given by entmax.entmax_bisect.
	"""
	if alpha == 1.0:
		return softmax
	elif alpha == 2.0:
		return sparsemax
	elif alpha == 1.5:
		return entmax15
	elif alpha < 1:
		raise NotImplementedError
	else:
		return partial(entmax_bisect, alpha=alpha)

