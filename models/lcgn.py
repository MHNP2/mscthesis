"""
Language Conditioned Graph Network for VQA.
https://arxiv.org/abs/1905.04405
Based on https://github.com/ronghanghu/lcgn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import pickle
import nltk
from copy import deepcopy

from activations import alpha_entmax
from data import BaseDataModule, CLEVRDataset
from utils import (
	clevr_collate_fn,
	ExponentialMovingAverage,
	create_dataset_splits,
)
from models.mac import MACCallback, MACEarlyStopping


def get_positional_encoding(H, W, dim=128):
	assert dim % 4 == 0
	c_period = 10000 ** np.linspace(0., 1., dim // 4)
	h_vec = np.tile(np.arange(0, H).reshape((H, 1, 1)), (1, W, 1)) / c_period
	w_vec = np.tile(np.arange(0, W).reshape((1, W, 1)), (H, 1, 1)) / c_period
	pos_enc = np.concatenate(
		(np.sin(h_vec), np.cos(h_vec), np.sin(w_vec), np.cos(w_vec)), axis=-1
	)
	return pos_enc.reshape((1, H, W, dim))


class Linear(nn.Linear):
	""" Linear layer with custom init """
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		fan_avg = (self.in_features + self.out_features) / 2.
		bound = np.sqrt(3. / fan_avg)
		nn.init.uniform_(self.weight, -bound, bound)
		if self.bias is not None:
			nn.init.constant_(self.bias, 0.)


def apply_mask1d(attention, image_locs):
	""" 1-dimensional attention mask """
	batch_size, num_loc = attention.size()
	tmp1 = attention.new_zeros(num_loc)
	tmp1[:num_loc] = torch.arange(
		0, num_loc, dtype=attention.dtype).unsqueeze(0)

	tmp1 = tmp1.expand(batch_size, num_loc)
	tmp2 = image_locs.type(tmp1.type())
	tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
	mask = torch.ge(tmp1, tmp2)
	attention = attention.masked_fill(mask, -1e30)
	return attention


def apply_mask2d(attention, image_locs):
	""" 2-dimensional attention mask """
	batch_size, num_loc, _ = attention.size()
	tmp1 = attention.new_zeros(num_loc)
	tmp1[:num_loc] = torch.arange(
		0, num_loc, dtype=attention.dtype).unsqueeze(0)

	tmp1 = tmp1.expand(batch_size, num_loc)
	tmp2 = image_locs.type(tmp1.type())
	tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
	mask1d = torch.ge(tmp1, tmp2)
	mask2d = mask1d[:, None, :] | mask1d[:, :, None]
	attention = attention.masked_fill(mask2d, -1e30)
	return attention


def generate_scaled_var_drop_mask(shape, keep_prob):
	""" Scaled dropout mask """
	assert keep_prob > 0. and keep_prob <= 1.
	mask = torch.rand(shape, device='cuda').le(keep_prob)
	mask = mask.float() / keep_prob
	return mask


class Encoder(nn.Module):
	"""Encoder for LCGN model. Embedding + Bidirectional LSTM."""
	def __init__(self, embInit, emb_dim, enc_dim,
		forget_gate_bias=1., encInputDropout=0.8, qDropout=0.92):
		super().__init__()
		self.embeddingsVar = nn.Parameter(
			torch.Tensor(embInit))
		self.emb_dim = embInit.shape[1]
		self.enc_input_drop = nn.Dropout(1 - encInputDropout)
		self.rnn0 = BiLSTM(emb_dim, enc_dim, forget_gate_bias=1.)
		self.question_drop = nn.Dropout(1 - qDropout)

	def forward(self, qIndices, questionLengths):
		# Word embedding
		embeddings = torch.cat(
			[torch.zeros(1, self.emb_dim, device=qIndices.device), self.embeddingsVar],
			dim=0)
		questions = F.embedding(qIndices, embeddings)
		questions = self.enc_input_drop(questions)

		# RNN (LSTM)
		questionCntxWords, vecQuestions = self.rnn0(questions, questionLengths)
		vecQuestions = self.question_drop(vecQuestions)

		return questionCntxWords, vecQuestions


class BiLSTM(nn.Module):
	""" Bidirectional LSTM with custom init. """
	def __init__(self, emb_dim, enc_dim, forget_gate_bias=1.):
		super().__init__()
		self.enc_dim = enc_dim
		self.emb_dim = emb_dim
		self.bilstm = torch.nn.LSTM(
			input_size=emb_dim, hidden_size=enc_dim // 2,
			num_layers=1, batch_first=True, bidirectional=True)
		self.forget_gate_bias = forget_gate_bias
		self.init_weights()

	def init_weights(self):
		forget_gate_bias = self.forget_gate_bias

		d = self.enc_dim // 2

		# initialize LSTM weights (to be consistent with TensorFlow)
		fan_avg = (d*4 + (d+self.emb_dim)) / 2.
		bound = np.sqrt(3. / fan_avg)
		nn.init.uniform_(self.bilstm.weight_ih_l0, -bound, bound)
		nn.init.uniform_(self.bilstm.weight_hh_l0, -bound, bound)
		nn.init.uniform_(self.bilstm.weight_ih_l0_reverse, -bound, bound)
		nn.init.uniform_(self.bilstm.weight_hh_l0_reverse, -bound, bound)

		# initialize LSTM forget gate bias (to be consistent with TensorFlow)
		self.bilstm.bias_ih_l0.data[...] = 0.
		self.bilstm.bias_ih_l0.data[d:2*d] = forget_gate_bias
		self.bilstm.bias_hh_l0.data[...] = 0.
		self.bilstm.bias_hh_l0.requires_grad = False
		self.bilstm.bias_ih_l0_reverse.data[...] = 0.
		self.bilstm.bias_ih_l0_reverse.data[d:2*d] = forget_gate_bias
		self.bilstm.bias_hh_l0_reverse.data[...] = 0.
		self.bilstm.bias_hh_l0_reverse.requires_grad = False

	def forward(self, questions, questionLengths):

		# pack questions for LSTM forwarding
		b, l, _ = questions.shape
		packed_questions = nn.utils.rnn.pack_padded_sequence(
			questions, questionLengths, batch_first=True)
		packed_output, (h_n, _) = self.bilstm(packed_questions)
		packed_output, _ = nn.utils.rnn.pad_packed_sequence(
			packed_output, batch_first=True, total_length=l)
		h_n = torch.transpose(h_n, 1, 0).reshape(b, -1)
		return packed_output, h_n


class Classifier(nn.Module):
	""" 2-layer classifier for LCGN. """
	def __init__(self, d_model, d_ctx, classifier_dim, num_choices, outputDropout=0.85):
		super().__init__()
		self.outQuestion = Linear(d_model, d_ctx)
		in_dim = 3 * d_ctx
		self.classifier_layer = nn.Sequential(
			nn.Dropout(1 - outputDropout),
			Linear(in_dim, classifier_dim),
			nn.ELU(),
			nn.Dropout(1 - outputDropout),
			Linear(classifier_dim, num_choices))

	def forward(self, x_att, vecQuestions):
		eQ = self.outQuestion(vecQuestions)
		features = torch.cat([x_att, eQ, x_att*eQ], dim=-1)
		logits = self.classifier_layer(features)
		return logits


class SingleHop(nn.Module):
	""" Single-hop attention module. """
	def __init__(self, enc_dim, d_ctx, alpha=1.0):
		super().__init__()
		self.proj_q = Linear(enc_dim, d_ctx)
		self.inter2att = Linear(d_ctx, 1)
		self.activ = alpha_entmax(alpha)

	def forward(self, kb, vecQuestions, imagesObjectNum):
		proj_q = self.proj_q(vecQuestions)
		interactions = F.normalize(kb * proj_q[:, None, :], dim=-1)
		raw_att = self.inter2att(interactions).squeeze(-1)
		raw_att = apply_mask1d(raw_att, imagesObjectNum)
		att = self.activ(raw_att, dim=-1)

		x_att = torch.bmm(att[:, None, :], kb).squeeze(1)
		return x_att


class LCGN(nn.Module):
	""" Main LCGN block: message passing. """
	def __init__(self, d_feat, d_ctx, d_model, num_steps,
		stemDropout=1.0, readDropout=0.85, memoryDropout=0.85,
		txt_alpha=1.0, msg_alpha=1.0):
		super().__init__()
		self.num_steps = num_steps
		self.d_ctx = d_ctx
		self.memoryDropout = memoryDropout
		self.build_loc_ctx_init(d_feat, d_ctx, stemDropout=stemDropout)
		self.build_extract_textual_command(d_model, num_steps, alpha=txt_alpha)
		self.build_propagate_message(d_ctx, readDropout=readDropout, alpha=msg_alpha)

	def build_loc_ctx_init(self, d_feat, d_ctx, stemDropout=1.0):
		self.initKB = Linear(d_feat, d_ctx)
		self.x_loc_drop = nn.Dropout(1 - stemDropout)
		self.initMem = nn.Parameter(torch.randn(1, 1, d_ctx))

	def build_extract_textual_command(self, d_model, num_steps, alpha=1.0):
		self.qInput = Linear(d_model, d_model)
		for t in range(num_steps):
			qInput_layer2 = Linear(d_model, d_model)
			setattr(self, "qInput%d" % t, qInput_layer2)
		self.cmd_inter2logits = Linear(d_model, 1)
		self.text_activ = alpha_entmax(alpha)

	def build_propagate_message(self, d_ctx, readDropout=0.85, alpha=1.0):
		self.read_drop = nn.Dropout(1 - readDropout)
		self.project_x_loc = Linear(d_ctx, d_ctx)
		self.project_x_ctx = Linear(d_ctx, d_ctx)
		self.queries = Linear(3*d_ctx, d_ctx)
		self.keys = Linear(3*d_ctx, d_ctx)
		self.vals = Linear(3*d_ctx, d_ctx)
		self.proj_keys = Linear(d_ctx, d_ctx)
		self.proj_vals = Linear(d_ctx, d_ctx)
		self.mem_update = Linear(2*d_ctx, d_ctx)
		self.combine_kb = Linear(2*d_ctx, d_ctx)
		self.msg_activ = alpha_entmax(alpha)

	def forward(self, images, q_encoding, lstm_outputs, batch_size, q_length,
				entity_num):
		x_loc, x_ctx, x_ctx_var_drop = self.loc_ctx_init(images)
		for t in range(self.num_steps):
			x_ctx = self.run_message_passing_iter(
				q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
				x_ctx_var_drop, entity_num, t)
		x_out = self.combine_kb(torch.cat([x_loc, x_ctx], dim=-1))
		return x_out

	def extract_textual_command(self, q_encoding, lstm_outputs, q_length, t):
		qInput_layer2 = getattr(self, "qInput%d" % t)
		act_fun = F.elu
		q_cmd = qInput_layer2(act_fun(self.qInput(q_encoding)))
		raw_att = self.cmd_inter2logits(
			q_cmd[:, None, :] * lstm_outputs).squeeze(-1)
		raw_att = apply_mask1d(raw_att, q_length)
		att = self.text_activ(raw_att, dim=-1)
		cmd = torch.bmm(att[:, None, :], lstm_outputs).squeeze(1)
		return cmd

	def propagate_message(self, cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num):
		x_ctx = x_ctx * x_ctx_var_drop
		proj_x_loc = self.project_x_loc(self.read_drop(x_loc))
		proj_x_ctx = self.project_x_ctx(self.read_drop(x_ctx))
		x_joint = torch.cat(
			[x_loc, x_ctx, proj_x_loc * proj_x_ctx], dim=-1)

		queries = self.queries(x_joint)
		keys = self.keys(x_joint) * self.proj_keys(cmd)[:, None, :]
		vals = self.vals(x_joint) * self.proj_vals(cmd)[:, None, :]
		edge_score = (
			torch.bmm(queries, torch.transpose(keys, 1, 2)) /
			np.sqrt(self.d_ctx))
		edge_score = apply_mask2d(edge_score, entity_num)
		edge_prob = self.msg_activ(edge_score, dim=-1)
		message = torch.bmm(edge_prob, vals)

		x_ctx_new = self.mem_update(torch.cat([x_ctx, message], dim=-1))
		return x_ctx_new

	def run_message_passing_iter(
			self, q_encoding, lstm_outputs, q_length, x_loc, x_ctx,
			x_ctx_var_drop, entity_num, t):
		cmd = self.extract_textual_command(
				q_encoding, lstm_outputs, q_length, t)
		x_ctx = self.propagate_message(
			cmd, x_loc, x_ctx, x_ctx_var_drop, entity_num)
		return x_ctx

	def loc_ctx_init(self, images):
		images = F.normalize(images, dim=-1)
		x_loc = self.initKB(images)
		x_loc = self.x_loc_drop(x_loc)
		x_ctx = self.initMem.expand(x_loc.size())
		x_ctx_var_drop = generate_scaled_var_drop_mask(
			x_ctx.size(),
			keep_prob=(self.memoryDropout if self.training else 1.))
		return x_loc, x_ctx, x_ctx_var_drop


class LCGNnet(nn.Module):
	""" Full LCGN model. """
	def __init__(self, num_vocab, num_choices, emb_dim, enc_dim, d_feat,
		d_ctx, d_model, num_steps, classifier_dim, stemDropout=1.0,
		readDropout=0.85, memoryDropout=0.85, forget_gate_bias=1.,
		encInputDropout=0.8, qDropout=0.92, outputDropout=0.85,
		txt_alpha=1.0, msg_alpha=1.0):
		super().__init__()
		embInit = np.random.uniform(
			low=-1, high=1, size=(num_vocab-1, emb_dim))
		self.num_vocab = num_vocab
		self.num_choices = num_choices
		self.encoder = Encoder(embInit, emb_dim, enc_dim, forget_gate_bias=forget_gate_bias,
			encInputDropout=encInputDropout, qDropout=qDropout)
		self.lcgn = LCGN(d_feat, d_ctx, d_model, num_steps, stemDropout=stemDropout,
			readDropout=readDropout, memoryDropout=memoryDropout, txt_alpha=txt_alpha,
			msg_alpha=msg_alpha)
		self.single_hop = SingleHop(enc_dim, d_ctx)
		self.classifier = Classifier(d_model, d_ctx, classifier_dim, num_choices,
			outputDropout=outputDropout)

	def forward(self, img, q, q_len, a):
	
		batchSize = img.size(0)
		images = img
		imagesObjectNum = torch.ones(batchSize) * img.size(1)
		answerIndices = a
		questionIndices = q
		questionLengths = q_len

		# LSTM
		questionCntxWords, vecQuestions = self.encoder(
			questionIndices, questionLengths)

		# LCGN
		x_out = self.lcgn(
			images=images, q_encoding=vecQuestions,
			lstm_outputs=questionCntxWords, batch_size=batchSize,
			q_length=questionLengths, entity_num=imagesObjectNum)

		# Single-Hop
		loss = torch.tensor(0., device=x_out.device)
		res = {}

		x_att = self.single_hop(x_out, vecQuestions, imagesObjectNum)
		logits = self.classifier(x_att, vecQuestions)
		predictions, num_correct = self.add_pred_op(logits, answerIndices)
		loss += self.add_answer_loss_op(logits, answerIndices)
		res.update({
			"predictions": predictions,
			"num_correct": int(num_correct),
			"accuracy": float(num_correct * 1. / batchSize)
		})

		res.update({"batch_size": int(batchSize), "loss": loss})
		return res

	def add_pred_op(self, logits, answers):
		preds = torch.argmax(logits, dim=-1).detach()
		corrects = (preds == answers)
		correctNum = torch.sum(corrects).item()
		preds = preds.cpu().numpy()

		return preds, correctNum

	def add_answer_loss_op(self, logits, answers):
		loss = F.cross_entropy(logits, answers)
		return loss


def collate_fn(batch):
	""" Collate function for LCGN. """
	img, q, q_len, a, _, _ = clevr_collate_fn(batch)
	q_len = torch.tensor(q_len)
	return img, q, q_len, a 


class LCGNDataModule(BaseDataModule):
	""" LCGN on CLEVR. """

	def __init__(self, path, batch_size, p=1.0, ood=False):
		super().__init__(path, batch_size, collate_fn=collate_fn)

		with open(path+f'/dic.pkl', 'rb') as f:
			dic = pickle.load(f)

		self.vocab = dic['vocab']
		self.answers = dic['answers']
		self.p = p
		self.ood = ood

		if self.ood:
			self.ood_idxs = load_json('ood-15.json')['excluded']

	def prepare_data(self):
		nltk.download('punkt')

	def setup(self, stage='fit'):

		if stage == 'fit':
			# load train set
			self.train_dataset = CLEVRDataset(self.path, split='train')
			if self.p < 1.0:
				self.train_dataset.idxs = create_dataset_splits(
					len(self.train_dataset), self.p)
			if self.ood:
				if self.train_dataset.idxs is None:
					idxs = list(range(len(self.train_dataset)))
				else:
					idxs = self.train_dataset.idxs
				self.train_dataset.idxs = list(set(idxs).difference(set(self.ood_idxs)))

			# load val set
			self.val_dataset = CLEVRDataset(self.path, split='val')
		
		else:
			# load test set
			self.test_dataset = CLEVRDataset(self.path, split='test')


class LCGNModule(pl.LightningModule):
	""" Lightning module for LCGN. """

	def __init__(self, num_vocab=None, num_choices=None, emb_dim=300, enc_dim=512,
		d_feat=512, d_ctx=512, d_model=512, num_steps=4, pe_dim=128, classifier_dim=512,
		stemDropout=1.0, readDropout=0.85, memoryDropout=0.85, forget_gate_bias=1.,
		encInputDropout=0.8, qDropout=0.92, outputDropout=0.85, lr=3e-4, decay=0.999,
		max_img_size=15, txt_alpha=1.0, msg_alpha=1.0):
		super().__init__()
		self.model = LCGNnet(num_vocab, num_choices, emb_dim, enc_dim,
			d_feat, d_ctx, d_model, num_steps, classifier_dim, stemDropout=stemDropout,
			readDropout=readDropout, memoryDropout=memoryDropout,
			forget_gate_bias=forget_gate_bias, encInputDropout=encInputDropout,
			qDropout=qDropout, outputDropout=outputDropout, txt_alpha=txt_alpha,
			msg_alpha=msg_alpha)
		self.register_buffer('pos_enc',
			torch.from_numpy(get_positional_encoding(max_img_size, max_img_size, dim=pe_dim)).float(),
		)
		self.ema = ExponentialMovingAverage(self.model.state_dict(), decay)
		self.lr = lr

	def shared_step(self, batch, batch_idx):
		img, q, q_len, a = batch 
		q_len = q_len.cpu()
		"""
		img : (batch, img_feat, H, W)
		q : (batch, seq_len)
		q_len : (batch,)
		a : (batch,)
		"""
		b, C, H, W = img.shape
		pos_enc = self.pos_enc[:,:H,:W,:].repeat(b, 1, 1, 1)
		img = torch.cat([img.permute(0, 2, 3, 1), pos_enc], dim=-1) # (b, H, W, C+pos)
		img = img.view(-1, H*W, img.size(-1)) # (batch, N, C+pos)

		res = self.model(img, q, q_len, a)
		loss, acc = res['loss'], res['accuracy']
		return loss, acc

	def training_step(self, batch, batch_idx):
		loss, acc = self.shared_step(batch, batch_idx)
		self.log('loss', loss)
		self.log('acc', acc, prog_bar=True)
		return loss

	def validation_step(self, batch, batch_idx):
		loss, acc = self.shared_step(batch, batch_idx)
		self.log('val_loss', loss)
		self.log('val_acc', acc)

	def test_step(self, batch, batch_idx):
		loss, acc = self.shared_step(batch, batch_idx)
		self.log('test_loss', loss)
		self.log('test_acc', acc)

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


# identical callback
LCGNCallback = MACCallback
LCGNEarlyStopping = MACEarlyStopping

