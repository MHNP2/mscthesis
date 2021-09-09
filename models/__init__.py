import torch 
import torch.nn as nn
from utils import regular_collate_fn


def init_model(params):
	"""
	Initiates the LightningModule and DataModule
	for the given model.
	"""
	if params.alpha != 'none':
		params.msg_alpha = params.alpha
		params.txt_alpha = params.alpha
		params.r_alpha = params.alpha
		params.c_alpha = params.alpha
		params.w_alpha = params.alpha

	callbacks = []
	if params.model == 'mac':
		from models.mac import (
			MACModule,
			MACDataModule,
			MACCallback,
			MACEarlyStopping,
		)
		dm = MACDataModule(
			params.root,
			params.batch_size,
			p=params.train_proportion,
			ood=params.ood,
		)
		dm.prepare_data()
		dm.setup(stage=params.stage)
		model = MACModule(
			n_vocab=len(dm.vocab),
			dim=params.d_model,
			embed_hidden=params.emb_dim,
			max_step=params.max_step,
			self_attention=params.self_attention,
			memory_gate=params.memory_gate,
			classes=params.classes,
			dropout=params.dropout,
			lr=params.lr,
			decay=params.decay,
			r_alpha=params.r_alpha,
			c_alpha=params.c_alpha,
			w_alpha=params.w_alpha,
			ood=dm.ood_val_idxs if params.ood is not None else None,
			ood_families=dm.ood_families if params.ood=='ood-15' else None,
		)
		callbacks.extend([
			MACCallback(),
			MACEarlyStopping(patience=params.patience),
		])
	elif params.model == 'lcgn':
		from models.lcgn import (
			LCGNModule,
			LCGNDataModule,
			LCGNCallback,
			LCGNEarlyStopping,
		)
		dm = LCGNDataModule(
			params.root,
			params.batch_size,
			p=params.train_proportion,
			ood=params.ood,
		)
		dm.prepare_data()
		dm.setup(stage=params.stage)
		if params.stage == 'test' or params.checkpoint:
			model = MACModule.load_from_checkpoint(
				'checkpoint/checkpoint_'+params.name+'.ckpt'
			)
		else:
			model = LCGNModule(
				num_vocab=len(dm.vocab),
				num_choices=len(dm.answers),
				emb_dim=params.emb_dim,
				enc_dim=params.d_model,
				d_feat=params.img_feats,
				d_ctx=params.d_model,
				d_model=params.d_model,
				num_steps=params.max_step,
				pe_dim=params.pe_dim,
				classifier_dim=params.d_model,
				stemDropout=params.stemDropout,
				readDropout=params.readDropout,
				memoryDropout=params.memoryDropout,
				encInputDropout=params.encInputDropout,
				qDropout=params.qDropout,
				outputDropout=params.outputDropout,
				lr=params.lr,
				decay=params.decay,
				msg_alpha=params.msg_alpha,
				txt_alpha=params.txt_alpha,
			)
		callbacks.extend([
			LCGNCallback(),
			LCGNEarlyStopping(patience=params.patience)
		])
	else:
		raise NotImplementedError

	return dm, model, callbacks

