from abc import ABC, abstractmethod
import lightning.pytorch as pl


class ABCModel(ABC, pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.outputs = None # for storing model outputs
		self.scores = None # for storing metric evals

  
	def on_validation_start(self):
		self.outputs = None
		self.scores = None
		super().on_validation_start()
	
	def on_test_start(self):
		self.outputs = None
		self.scores = None
		super().on_test_start()
  
	def on_predict_start(self) -> None:
		self.outputs = None
		self.scores = None
		super().on_predict_start()
		
	@abstractmethod
	def reconstruct(self):
		"""For reconstructing model outputs"""
		pass
	
	@abstractmethod
	def compute_metrics(self):
		"""For computing loss, performance metrics, etc."""
		pass