""" Model Arguments """

from argparse import ArgumentParser
parser = ArgumentParser()

# required
parser.add_argument('--config',
	required=True, type=str,
	help='Name of config set.'
)
parser.add_argument('--root',
	required=True, type=str,
	help='CLEVR dataset directory.'
)
parser.add_argument('--name',
	required=True, type=str,
	help='Name of run.'
)

# general
parser.add_argument('--seed',
	default=None, type=int,
	help='Random seed for reproducibility.'
)
parser.add_argument('--wandb',
	default=None, type=str,
	help='Wandb entity for logging.'
)
parser.add_argument('--group',
	default=None, type=str,
	help='Wandb group for logging.'
)
parser.add_argument('--stage',
	default='fit', type=str,
	help='Training or testing.'
)
parser.add_argument('--checkpoint',
	action='store_true',
	help='Restore checkpoint.'
)
parser.add_argument('--sample',
	action='store_true',
	help='Use sample data.'
)
parser.add_argument('--train_proportion',
	default=None, type=float,
	help='Proportion of train data to use.'
)
parser.add_argument('--ood',
	default=None, type=str,
	help='Use out-of-distribution set.'
)

# training
parser.add_argument('--batch_size',
	default=None, type=int,
	help='Batch size'
)
parser.add_argument('--epochs',
	default=None, type=int,
	help='Number of training epochs.'
)
parser.add_argument('--lr',
	default=None, type=float,
	help='Learning rate.'
)
parser.add_argument('--gradient_clip',
	default=None, type=float,
	help='Max norm of gradients.'
)
parser.add_argument('--decay',
	default=None, type=float,
	help='EMA accumulation rate.'
)
parser.add_argument('--patience',
	default=None, type=int,
	help='Early stopping patience'
)

# model shared
parser.add_argument('--model',
	default=None, type=str,
	help='Name of model.'
)
parser.add_argument('--classes',
	default=None, type=int,
	help='Number of output classes'
)
parser.add_argument('--emb_dim',
	default=None, type=int,
	help='Embedding dimension.'
)
parser.add_argument('--img_feats',
	default=None, type=int,
	help='Image features.'
)
parser.add_argument('--d_model',
	default=None, type=int,
	help='Model dimension.'
)
parser.add_argument('--max_step',
	default=None, type=int,
	help='Number of memory/node updates.'
)
parser.add_argument('--alpha',
	default=None, type=float,
	help='Overwrite all alphas.')


# model specific
# MAC
parser.add_argument('--self_attention',
	action='store_true',
	help='Use self attention in write.'
)
parser.add_argument('--memory_gate',
	action='store_true',
	help='Use memory gate in write.'
)
parser.add_argument('--dropout',
	default=None, type=float,
	help='MAC dropout rate.'
)
parser.add_argument('--c_alpha',
	default=None, type=float,
	help='Activation function.'
)
parser.add_argument('--r_alpha',
	default=None, type=float,
	help='Activation function.'
)
parser.add_argument('--w_alpha',
	default=None, type=float,
	help='Activation function.'
)


# LCGN
parser.add_argument('--pe_dim',
	default=None, type=int,
	help='Dimension of PE.'
)
parser.add_argument('--stemDropout',
	default=None, type=float,
	help='Stem dropout.'
)
parser.add_argument('--readDropout',
	default=None, type=float,
	help='Read dropout.'
)
parser.add_argument('--memoryDropout',
	default=None, type=float,
	help='Memory dropout.'
)
parser.add_argument('--encInputDropout',
	default=None, type=float,
	help='Encoder input dropout.'
)
parser.add_argument('--qDropout',
	default=None, type=float,
	help='Question dropout.'
)
parser.add_argument('--outputDropout',
	default=None, type=float,
	help='Question dropout.'
)
parser.add_argument('--txt_alpha',
	default=None, type=float,
	help='Activation function.'
)
parser.add_argument('--msg_alpha',
	default=None, type=float,
	help='Activation function.'
)

