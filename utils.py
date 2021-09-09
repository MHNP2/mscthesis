""" Functions for loading data etc. """
import torch
import pytorch_lightning as pl
import numpy as np 
import nltk 
import json
import pickle


def load_json(path):
	with open(path, 'r') as f:
		data = json.load(f)
	return data


def load_pkl(path):
	with open(path, 'rb') as f:
		data = pickle.load(f)
	return data


def load_glove(path):
	word2vec = {}
	with open(path, 'r') as f:
		for line in f:
			line = line.split()
			w = line[0]
			vec = np.array(line[1:], dtype=np.float32)
			word2vec[w] = vec
	return word2vec


def load_vocab(path):
	vocab = load_json(path)
	word2idx = {w:i for i, w in enumerate(vocab)}
	return word2idx


def load_answers(path):
	vocab = load_json(path)
	answers = {a:i for i, a in enumerate(vocab)}
	classes = len(vocab)
	return answers, classes


def preprocess_question(q, vocab=None, word2vec=None):
	""" Preprocess a question by tokenizing, removing
	's and s', removing words not in an optional
	pretrained embedding set and then converting to idx."""
	q = nltk.tokenize.word_tokenize(q.lower())
	q_processed = []
	for q_ in q:
		if q_[-2:] == "'s":
			q_ = q_[:-2]
		elif q[:-2:] == "s'":
			q_ = q_[:-1]
		if word2vec is not None:
			if q_ in word2vec:
				q_processed.append(q_)
		else:
			q_processed.append(q_)
	if vocab is not None:
		q_processed = [vocab[q_] for q_ in q_processed if q_ in vocab]
	return q_processed


def word2vec2embed(word2vec, word2idx):
	""" Creates an embedding layer from a dictionary
	of pretrained embeddings and vocab dictionary."""
	emb_dim = word2vec['the'].shape[0]
	emb = torch.nn.Embedding(len(word2idx), emb_dim) 
	emb_matrix = []
	for w, idx in word2idx.items():
		if w in word2vec:
			emb_matrix.append(word2vec[w])
		else:
			emb_matrix.append(np.zeros(emb_dim,))
	emb.weight.data.copy_(torch.from_numpy(np.array(emb_matrix)))
	return emb


def regular_collate_fn(data):
	""" Collate of all object features for GQA. """
	img, box, q, a = list(zip(*data))
	q = torch.nn.utils.rnn.pad_sequence(q, batch_first=True)
	return torch.stack(img), torch.stack(box), q, torch.stack(a).long()


def clevr_collate_fn(data):
	""" Collate img features for CLEVR.
	Sort questions by descending length."""
	data = sorted(data, key=lambda x: len(x[1]), reverse=True)
	img, q, len_q, a, f, idx = list(zip(*data))
	q = torch.nn.utils.rnn.pad_sequence(q, batch_first=True)
	return torch.stack(img), q, list(len_q), torch.stack(a), list(f), list(idx)


def create_dataset_splits(n, p=1.0):
	""" Randomly permute a dataset of length
	n. Optionally select the first p% of the
	dataset. """
	perm = np.random.permutation(n).tolist()
	idx = int(p * n)
	return perm[:idx]


class ExponentialMovingAverage:
	""" Maintains a state-dict of an exponential moving average
	of model weights. """
	def __init__(self, param_dict, decay):
		assert decay >= 0. and decay < 1.
		self.decay = decay
		self.params_ema = {
			name: torch.zeros_like(p.data) for name, p in param_dict.items()}

	def step(self, param_dict):
		for name, p in param_dict.items():
			d = self.params_ema[name].device
			self.params_ema[name].mul_(self.decay).add_(1-self.decay, p.data.to(d))

	@property
	def state_dict(self):
		return self.params_ema

	def load_state_dict(self, state_dict):
		assert self.params_ema.keys() == state_dict.keys()
		for k, p_ema in self.params_ema.items():
			assert state_dict[k].dtype == p_ema.dtype
			assert state_dict[k].size() == p_ema.size()
			p_ema[...] = state_dict[k]

	def set_params_from_ema(self, param_dict):
		for k, p in param_dict.items():
			p.data[...] = self.params_ema[k]

