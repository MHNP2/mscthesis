""" Train a model on a dataset."""
import torch 
import pytorch_lightning as pl 
import numpy as np 

from models import init_model


def main(params):

	pl.seed_everything(params.seed)

	# logging
	if params.wandb != 'none':
		logger = pl.loggers.WandbLogger(name=params.name, project='Sparse-VQA',
			entity=params.wandb, group=params.group,
			config={
			k: v for k, v in params.__dict__.items() if isinstance(v, (float, int, str, list))
			})
	else:
		logger = True

	# model and datamodule
	dm, model, callbacks = init_model(params)
	if params.stage == 'test' or params.checkpoint:
		model._load_state_dict('checkpoint/' + params.name)

	# trainer setup
	gpus = -1 if torch.cuda.is_available() else None 
	trainer = pl.Trainer(gpus=gpus,
		max_epochs=params.epochs,
		deterministic=True,
		logger=logger,
		callbacks=callbacks,
		gradient_clip_val=params.gradient_clip)

	# run
	if params.stage == 'fit':
		trainer.fit(model, dm)
		torch.save(model.state_dict(), 'checkpoint/' + params.name)
		trainer.test(model, dm.test_dataloader())
	elif params.stage == 'test':
		trainer.test(model, dm.test_dataloader())


if __name__ == '__main__':
	from loader import Loader 
	from arguments import parser
	params = Loader(parser.parse_args())
	main(params)

