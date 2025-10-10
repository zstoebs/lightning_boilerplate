from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Literal, List, Callable, Mapping
from inspect import isfunction
import lightning.pytorch as pl

from ..losses.ABCLoss import ABCLoss
from ..losses.MSE import NRMSELoss


class ABCModel(ABC, pl.LightningModule):
	def __init__(self, 
              loss_fn: nn.Module, 
              metrics: List[Callable[...,torch.Tensor]]=[NRMSELoss], 
              **kwargs):
		super().__init__()
		self.loss_fn = loss_fn
		self.metrics = metrics
		self.iscustom = isinstance(loss_fn, ABCLoss)
		self.outputs = [] # list for storing model outputs
		self.scores = {} # for storing metric evals

	def on_validation_start(self):
		self.outputs = []
		self.scores = {}
		super().on_validation_start()
	
	def on_test_start(self):
		self.outputs = []
		self.scores = {}
		super().on_test_start()
  
	def on_predict_start(self) -> None:
		self.outputs = []
		self.scores = {}
		super().on_predict_start()
	
	def training_step(self, batch, batch_idx: int=0, **kwargs) -> torch.Tensor:
		if self.iscustom:
			inputs = self.loss_fn.prepare_input(**batch) 
			out = self(**inputs)
			outputs = self.loss_fn.prepare_output(**out, **inputs)
		else:
			out = self(**batch)
			outputs = {**out, **batch}

		scores = self.compute_metrics(**outputs, stage='train')
		self.log_dict(scores, sync_dist=True)

		loss = scores['train_loss']
		return loss

	def validation_step(self, batch, batch_idx: int=0, **kwargs) -> torch.Tensor:
		if self.iscustom:
			inputs = self.loss_fn.prepare_input(**batch) 
			out = self(**inputs)
			outputs = self.loss_fn.prepare_output(**out, **inputs)
		else:
			out = self(**batch)
			outputs = {**out, **batch}

		self.outputs += [{key: val.detach().cpu().numpy().copy() for key, val in outputs.items()}]

		scores = self.compute_metrics(**outputs, stage='val')
		self.scores = {key: val.cpu() for key, val in scores.items()} 
		self.log_dict(scores, sync_dist=True)

		loss = scores['val_loss']
		return loss

	def test_step(self, batch, batch_idx: int=0, **kwargs) -> torch.Tensor:
		if self.iscustom:
			inputs = self.loss_fn.prepare_input(**batch) 
			out = self(**inputs)
			outputs = self.loss_fn.prepare_output(**out, **inputs)
		else:
			out = self(**batch)
			outputs = {**out, **batch}

		self.outputs += [{key: val.detach().cpu().numpy().copy() for key, val in outputs.items()}]

		scores = self.compute_metrics(**outputs, stage='test')
		self.scores = {key: val.cpu() for key, val in scores.items()} 
		self.log_dict(scores, sync_dist=True)
		
		loss = scores['test_loss']
		return loss
	
	def predict_step(self, batch, batch_idx: int=0, **kwargs) -> torch.Tensor:
		if self.iscustom:
			inputs = self.loss_fn.prepare_input(**batch) 
			out = self(**inputs)
			outputs = self.loss_fn.prepare_output(**out, **inputs)
		else:
			out = self(**batch)
			outputs = {**out, **batch} 
			
		self.outputs += [{key: val.detach().cpu().numpy().copy()for key, val in outputs.items()}]
		
		scores = self.compute_metrics(**outputs, stage='predict')
		self.scores = {key: val.cpu() for key, val in scores.items()} 

		return self.outputs

	def compute_metrics(self, stage: Literal['train', 'val', 'test', 'predict'], **kwargs) -> Mapping[str, torch.Tensor]:
		scores = {}
		
		loss = self.loss_fn(**kwargs) # if not stage == 'predict' else torch.tensor(0.)
		scores[f"{stage}_loss"] = loss

		for metric in self.metrics:
			name = metric.__name__ if isfunction(metric) else metric.__class__.__name__
			scores[f"{stage}_{name}"] = metric(**kwargs)
		
		return scores
		
	@abstractmethod
	def reconstruct(self):
		"""For reconstructing model outputs"""
		pass
	
