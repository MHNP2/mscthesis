"""
Memory, Attention and Control (MAC) network. 
From https://arxiv.org/abs/1803.03067.
Implentation based on https://github.com/rosinality/mac-network-pytorch
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import pytorch_lightning as pl 
import numpy as np 
import pickle
import nltk
from copy import deepcopy
from torch.nn.init import kaiming_uniform_, xavier_uniform_, normal

from activations import alpha_entmax
from data import BaseDataModule, CLEVRDataset
from utils import (
	load_json,
	clevr_collate_fn,
	word2vec2embed,
	load_glove,
	create_dataset_splits,
	ExponentialMovingAverage,
)

class ControlUnit(nn.Module):
	
	def __init__(self, dim, max_step, alpha=1.0):
		super().__init__()
		self.dim = dim
		self.position_aware = nn.ModuleList(
			[nn.Linear(2*dim, dim) for _ in range(max_step)]
		)
		self.control_question = nn.Linear(2*dim, dim)
		self.attn = nn.Linear(dim, 1)
		self.activ = alpha_entmax(alpha)

	def forward(self, t, ctx, q, c):
		"""
		t : timestep
		ctx : contextual question
		q : atomic question
		c : control vectors
		"""
		p = self.position_aware[t](q)

		c = torch.cat([c, p], dim=1)
		c = self.control_question(c).unsqueeze(1)

		ctx_p = c * ctx
		attn_weight = self.activ(self.attn(ctx_p), dim=1)

		return (attn_weight * ctx).sum(1)


class ReadUnit(nn.Module):

	def __init__(self, dim, alpha=1.0):
		super().__init__()
		self.dim = dim
		self.mem = nn.Linear(dim, dim)
		self.concat = nn.Linear(2*dim, dim)
		self.attn = nn.Linear(dim, 1)
		self.activ = alpha_entmax(alpha)

	def forward(self, m, k, c):
		"""
		m : current memory
		k : knowledge base
		c : control vector
		"""
		m = self.mem(m[-1]).unsqueeze(2)
		concat = self.concat(torch.cat([m * k, k], 1).permute(0,2,1))
		attn_logit = self.attn(concat * c[-1].unsqueeze(1)).squeeze(2)
		attn_weight = self.activ(attn_logit, dim=1).unsqueeze(1)
		r = (attn_weight * k).sum(2)
		return r, attn_weight


class WriteUnit(nn.Module):

	def __init__(self, dim, self_attention=False, memory_gate=False, alpha=1.0):
		super().__init__()
		self.dim = dim

		self.concat = nn.Linear(2*dim, dim)

		self.self_attention = self_attention
		if self_attention:
			self.attn = nn.Linear(dim, 1)
			self.mem = nn.Linear(dim, dim)
			self.activ = alpha_entmax(alpha)

		self.memory_gate = memory_gate
		if memory_gate:
			self.control = nn.Linear(dim, 1)

	def forward(self, m, r, c):
		"""
		m : memory
		r : retrieved by read unit
		c : control
		"""
		m_prev = m[-1]
		concat = self.concat(torch.cat([r, m_prev], dim=1))
		next_m = concat

		if self.self_attention:
			c_cat = torch.stack(c[:-1], 2)
			attn = c[-1].unsqueeze(2) * c_cat
			attn = self.attn(attn.permute(0, 2, 1))
			attn = self.activ(attn, dim=1).permute(0, 2, 1)

			m_cat = torch.stack(m, dim=2)
			attn_m = (attn * m_cat).sum(2)
			next_m = self.mem(attn_m) + concat

		if self.memory_gate:
			c = self.control(c[-1])
			gate = F.sigmoid(c)
			next_m = gate * m_prev + (1.0 - gate) * next_m

		return next_m


class MACUnit(nn.Module):

	def __init__(self, dim, max_step=12, self_attention=False,
		memory_gate=False, dropout=0.15, c_alpha=1.0, r_alpha=1.0,
		w_alpha=1.0):
		super().__init__()

		self.control = ControlUnit(dim, max_step, alpha=c_alpha)
		self.read = ReadUnit(dim, alpha=r_alpha)
		self.write = WriteUnit(dim, self_attention, memory_gate, w_alpha)

		self.mem_0 = nn.Parameter(torch.zeros(1, dim))
		self.control_0 = nn.Parameter(torch.zeros(1, dim))

		self.dim = dim 
		self.max_step = max_step
		self.dropout = dropout

	def get_mask(self, x, dropout):
		mask = torch.empty_like(x).bernoulli_(1.0 - dropout)
		mask = mask / (1.0 - dropout)
		return mask 

	def forward(self, ctx, q, k):
		b_size = q.size(0)

		c = self.control_0.expand(b_size, self.dim)
		m = self.mem_0.expand(b_size, self.dim)

		if self.training:
			c_mask = self.get_mask(c, self.dropout)
			m_mask = self.get_mask(m, self.dropout)
			c = c * c_mask 
			m = m * m_mask

		controls = [c]
		memories = [m]

		for i in range(self.max_step):
			# control
			c = self.control(i, ctx, q, c)
			if self.training:
				c = c * c_mask
			controls.append(c)

			# read and write
			r, _ = self.read(memories, k, controls)
			m = self.write(memories, r, controls)
			if self.training:
				m = m * m_mask
			memories.append(m)

		return m


class MACNetwork(nn.Module):

	def __init__(self, n_vocab=None, dim=512, embed_hidden=300,
		max_step=12, self_attention=False, memory_gate=False,
		classes=28, dropout=0.15, c_alpha=1.0, r_alpha=1.0,
		w_alpha=1.0):
		super().__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(1024, dim, 3, padding=1),
			nn.ELU(),
			nn.Conv2d(dim, dim, 3, padding=1),
			nn.ELU(),
		)

		self.embed = nn.Embedding(n_vocab, embed_hidden)
		self.lstm = nn.LSTM(embed_hidden, dim,
			batch_first=True, bidirectional=True)
		self.lstm_proj = nn.Linear(2*dim, dim)

		self.mac = MACUnit(dim, max_step, self_attention,
			memory_gate, dropout, c_alpha, r_alpha, w_alpha)

		self.classifier = nn.Sequential(
			nn.Linear(3*dim, dim),
			nn.ELU(),
			nn.Linear(dim, classes)
		)

		self.max_step = max_step
		self.dim = dim 
		self.init_weights()

	def init_weights(self):
		self.embed.weight.data.uniform_(0, 1)

		kaiming_uniform_(self.conv[0].weight)
		self.conv[0].bias.data.zero_()
		kaiming_uniform_(self.conv[2].weight)
		self.conv[2].bias.data.zero_()

		kaiming_uniform_(self.classifier[0].weight)

		def xavier_init(m):
			if type(m) == nn.Linear:
				xavier_uniform_(m.weight)
				m.bias.data.zero_()
		self.mac.apply(xavier_init)

	def forward(self, img, q, q_len):
		b = img.size(0)
		img = self.conv(img)
		img = img.view(b, self.dim, -1)

		embed = self.embed(q)
		embed = nn.utils.rnn.pack_padded_sequence(embed, q_len,
    	batch_first=True)
		lstm_out, (h, _) = self.lstm(embed)
		lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, 
			batch_first=True)
		lstm_out = self.lstm_proj(lstm_out)
		h = h.permute(1, 0, 2).contiguous().view(b, -1)

		memory = self.mac(lstm_out, h, img)

		out = torch.cat([memory, h], 1)
		out = self.classifier(out)

		return out


class MACModule(pl.LightningModule):

	def __init__(self, lr=1e-4, decay=0.999, ood=None, ood_families=None, **kwargs):
		super().__init__()
		self.model = MACNetwork(**kwargs)
		self.criterion = nn.CrossEntropyLoss()
		self.lr = lr
		self.decay = decay
		self.ema = ExponentialMovingAverage(self.model.state_dict(), decay)
		self.ood = ood
		self.ood_families = ood_families

	def training_step(self, batch, batch_idx):
		img, q, q_len, a, _, _ = batch 
		loss, acc = self.shared_step(img, q, q_len, a)
		self.log('loss', loss)
		self.log('acc', acc.mean(), prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		img, q, q_len, a, f, idx = batch 
		loss, acc = self.shared_step(img, q, q_len, a)
		self.log('val_loss', loss)
		self.log('val_acc', acc.mean(), prog_bar=True)

		if self.ood_families is not None:
			ood = [f_ in self.ood_families for f_ in f]
			ood_acc = acc[ood].mean()
			if not torch.isnan(ood_acc):
				self.log('ood_acc', ood_acc, prog_bar=True)
		elif self.ood is not None:
			ood = [(i in self.ood) for i in idx]
			ood_acc = acc[ood].mean()
			if not torch.isnan(ood_acc):
				self.log('ood_acc', ood_acc, prog_bar=True)

	def test_step(self, batch, batch_idx):
		img, q, q_len, a, f, idx = batch 
		loss, acc = self.shared_step(img, q, q_len, a)
		self.log('test_loss', loss)
		self.log('test_acc', acc.mean(), prog_bar=True)

		if self.ood_families is not None:
			ood = [f_ in self.ood_families for f_ in f]
			ood_acc = acc[ood].mean()
			if not torch.isnan(ood_acc):
				self.log('ood_acc', ood_acc, prog_bar=True)
		elif self.ood is not None:
			ood = [(i in self.ood) for i in idx]
			ood_acc = acc[ood].mean()
			if not torch.isnan(ood_acc):
				self.log('ood_acc', ood_acc, prog_bar=True)

	def shared_step(self, img, q, q_len, labels):
		"""
		img : (batch, num_visual_features, visual_feat_dim)
		q : (batch, num_visual_features, visual_pos_dim)
		labels : (batch,)
		"""
		y_pred = self.model(img, q, q_len)
		loss = self.criterion(y_pred, labels)
		acc = (y_pred.argmax(-1) == labels).float()
		return loss, acc

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=self.lr)

	def _load_ema_dict(self):
		self.train_state_dict = deepcopy(self.model.state_dict())
		self.model.load_state_dict(self.ema.state_dict)

	def _load_train_dict(self):
		self.model.load_state_dict(self.train_state_dict)

	def _load_state_dict(self, path):
		state_dict = torch.load(path)
		self.load_state_dict(state_dict)
		self.ema.params_ema = self.model.state_dict()


class MACDataModule(BaseDataModule):
	""" MAC on CLEVR. """

	def __init__(self, path, batch_size, p=1.0, ood=False):
		super().__init__(path, batch_size, collate_fn=clevr_collate_fn)

		with open(path+f'/dic.pkl', 'rb') as f:
			dic = pickle.load(f)

		self.vocab = dic['vocab']
		self.answers = dic['answers']
		self.p = p
		self.ood = ood

		if self.ood is not None:
			f_ood = load_json(f'ood/{self.ood}.json')
			if self.ood == 'ood-15':
				self.ood_idxs = f_ood['excluded']
				self.ood_families = f_ood['families']
			else:
				self.ood_idxs = f_ood
				self.ood_val_idxs = load_json(f'ood/{self.ood}-val.json')

	def prepare_data(self):
		nltk.download('punkt')

	def setup(self, stage='fit'):

		if stage == 'fit':
			# load train set
			self.train_dataset = CLEVRDataset(self.path, split='train')
			if self.p < 1.0:
				self.train_dataset.idxs = create_dataset_splits(
					len(self.train_dataset), self.p)
			if self.ood is not None:
				if self.train_dataset.idxs is None:
					idxs = list(range(len(self.train_dataset)))
				else:
					idxs = self.train_dataset.idxs
				self.train_dataset.idxs = list(set(idxs).difference(set(self.ood_idxs)))

			# load val set
			self.val_dataset = CLEVRDataset(self.path, split='dev')
		
		else:
			# load test set
			self.test_dataset = CLEVRDataset(self.path, split='dev_test')


class MACCallback(pl.callbacks.Callback):
	""" Callback to accumulate val model. """
	def on_batch_end(self, trainer, pl_module):
		pl_module.ema.step(pl_module.model.state_dict())

	def on_validation_start(self, trainer, pl_module):
		pl_module._load_ema_dict()

	def on_validation_end(self, trainer, pl_module):
		pl_module._load_train_dict()

	def on_test_start(self, trainer, pl_module):
		pl_module._load_ema_dict()

	def on_test_end(self, trainer, pl_module):
		pl_module._load_train_dict()


class MACEarlyStopping(pl.callbacks.EarlyStopping):
	""" Early stopping for EMA dict. """
	def __init__(self, monitor='val_acc', patience=2, mode='max'):
		super().__init__(monitor=monitor, patience=patience, mode=mode, verbose=True)
		self.state_dict = None

	def on_validation_end(self, trainer, pl_module):
		if self._check_on_train_epoch_end or self._should_skip_check(trainer):
			return
		self._run_early_stopping_check(trainer)

		if self.wait_count == 0:
			self.state_dict = pl_module.ema.state_dict

		if trainer.should_stop:
			pl_module.model.load_state_dict(self.state_dict)
			pl_module.ema.load_state_dict(self.state_dict)

