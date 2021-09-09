""" Dataset and base classes. """
import torch
import pytorch_lightning as pl
import numpy as np
import pickle
import h5py
import os
from torchvision import transforms
from torchvision.models.resnet import ResNet, resnet101
from PIL import Image
from tqdm import tqdm

from utils import (
	load_json,
	load_pkl,
	load_vocab,
	load_answers,
	regular_collate_fn,
	clevr_collate_fn,
	preprocess_question,
)


class GQADataset(torch.utils.data.Dataset):
	""" GQA dataset with Faster-RCNN object features. """

	def __init__(self, path, questions, answers, tokenize_fn=None):
		super().__init__()
		self.path = path
		self.question_ids = list(questions.keys())
		self.questions = questions
		self.answers = answers
		self.tokenize_fn = tokenize_fn
		if self.tokenize_fn is None:
			self.word2idx = load_vocab(path+'/newqVocabFile.json')
			self.tokenize_fn = lambda q: torch.tensor(preprocess_question(q, self.word2idx))

	def __len__(self):
		return len(self.questions)

	def __getitem__(self, idx):

		# load question
		question = self.questions[self.question_ids[idx]]
		q, a = question['question'], question['answer']

		# answer to idx
		a = torch.tensor(self.answers[a], dtype=torch.int32)

		# question to idx
		q = self.tokenize_fn(q) # (seq_len,)

		# load image
		image_id = question['imageId']
		img = torch.load(self.path+'/features/{}_feature.pth'.format(image_id)) # (objects, feats)
		box = torch.load(self.path+'/features/{}_bbox.pth'.format(image_id)) # (objects, 4)
		
		return img, box, q, a


class CLEVRRaw(torch.utils.data.Dataset):
	""" CLEVR with raw images. """
	def __init__(self, root, transform, split='train'):
		self.root = root
		self.split = split
		self.transform = transform

	def __getitem__(self, idx):
		img = os.path.join(
			self.root, 'images', self.split, 'CLEVR_{}_{}.png'.format(
				self.split, str(idx).zfill(6)
			)
		)
		img = Image.open(img).convert('RGB')
		return self.transform(img)

	def __len__(self):
		return len(os.listdir(os.path.join(self.root,'images', self.split)))


class CLEVRDataset(torch.utils.data.Dataset):
	""" CLEVR dataset with ResNet101 features."""

	def __init__(self, root, split='train', idxs=None):
		self.data = load_pkl(root+f'/{split}.pkl')
		self.root = root
		self.split = split
		feat_split = 'val' if split == 'dev' or split == 'dev_test' else split
		self.h = h5py.File(root+f'/{feat_split}_features.hdf5', 'r')
		self.img = self.h['data']
		self.idxs = idxs

	def close(self):
		self.h.close()

	def __len__(self):
		return len(self.data) if self.idxs is None else len(self.idxs)

	def __getitem__(self, idx):
		idx = self.idxs[idx] if self.idxs is not None else idx
		img_file, q, a, f = self.data[idx]
		img_id = int(img_file.rsplit('_', 1)[1][:-4])
		img = torch.from_numpy(self.img[img_id])
		a = torch.tensor(a, dtype=torch.int64)
		return img, torch.LongTensor(q), len(q), a, f, idx	


class BaseDataModule(pl.LightningDataModule):
	""" Base data module for all models. """

	def __init__(self, path, batch_size, collate_fn=None, sample=False):
		super().__init__()
		self.path = path 
		self.batch_size = batch_size
		self.collate_fn = collate_fn
		self.sample = sample

	def prepare_data(self):
		raise NotImplementedError

	def setup(self, stage='fit'):
		raise NotImplementedError

	def train_dataloader(self):
		return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
			shuffle=False, collate_fn=self.collate_fn, num_workers=4,
			pin_memory=torch.cuda.is_available())

	def val_dataloader(self):
		return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size,
			shuffle=False, collate_fn=self.collate_fn, num_workers=4,
			pin_memory=torch.cuda.is_available())

	def test_dataloader(self):
		return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
			shuffle=False, collate_fn=self.collate_fn, num_workers=4,
			pin_memory=torch.cuda.is_available())


def preprocess_clevr_questions(path, split, vocab={'[PAD]':0}, answers={}):
	""" Build vocabularies and tokenize questions. """
	data = load_json(path+f'/questions/CLEVR_{split}_questions.json')

	result = []
	for question in data['questions']:
		words = preprocess_question(question['question'])
		q_tok = []
		for w in words:
			if w not in vocab:
				vocab[w] = len(vocab)
			q_tok.append(vocab[w])

		answer = question['answer']
		if answer not in answers:
			answers[answer] = len(answers)
		answer = answers[answer]

		result.append((question['image_filename'], q_tok, answer, question['question_family_index']))

	with open(path+f'/{split}.pkl', 'wb') as f:
		pickle.dump(result, f)

	with open(path+'/dic.pkl', 'wb') as f:
		pickle.dump({'vocab':vocab,'answers':answers}, f)


def preprocess_clevr_images(path, splits=['train', 'val'], batch_size=50):
	""" Extract ResNet101 features. """
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def forward(self, x):
		x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
		return self.layer3(self.layer2(self.layer1(x)))

	transform = transforms.Compose([
		transforms.Resize([224, 224]),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

	resnet = resnet101(True).to(device)
	resnet.eval()
	resnet.forward = forward.__get__(resnet, ResNet)

	for split in splits:

		dataloader = torch.utils.data.DataLoader(
			CLEVRRaw(path, transform, split=split),
			batch_size=batch_size, num_workers=4)
		f = h5py.File(path+'/{}_features.hdf5'.format(split), 'w', libver='latest')
		dset = f.create_dataset('data', (len(dataloader)*batch_size, 1024, 14, 14), dtype='f4')

		with torch.no_grad():
			for i, img in enumerate(dataloader):
				img = img.to(device)
				features = resnet(img).cpu().numpy()
				dset[i * batch_size:(i+1)*batch_size] = features

		f.close()


def preprocess_clevr(path):
	""" Preprocess the CLEVR dataset: build vocabularies,
	tokenize questions and extract image features. """

	preprocess_clevr_questions(path, 'train')
	dic = load_pkl(path+'/dic.pkl')
	vocab, answers = dic['vocab'], dic['answers']
	preprocess_clevr_questions(path, 'val',
		vocab=vocab, answers=answers)
	preprocess_clevr_images(path)


def build_val(path, seed=11, n=15000):
	""" Split validation data into a dev and dev-test
	set of a given size with a fixed seed. """
	data = load_pkl(path+'/val.pkl')
	print('Total validationd data:', len(data))

	np.random.seed(seed)
	dev_test = np.random.choice(len(data),
		size=(n,),
		replace=False)

	dev_test_set = []
	dev_set = []
	for i, x in tqdm(enumerate(data)):
		if i in dev_test:
			dev_test_set.append(x)
		else:
			dev_set.append(x)

	print('Dev-test size', len(dev_test_set))
	print('Dev size', len(dev_set))

	with open(path + '/dev.pkl', 'wb') as f:
		pickle.dump(dev_set, f)

	with open(path + '/dev_test.pkl', 'wb') as f:
		pickle.dump(dev_test_set, f)

