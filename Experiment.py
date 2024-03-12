import argparse
import numpy as np

import os
import flwr as fl
from flwr.common import NDArrays, Scalar
from programming_api.common import context, logger, backend, measures
from programming_api import Model, Dataset

class Experiment(fl.client.NumPyClient):
    def __init__(self, model : Model, dataset : Dataset, **kwargs) -> None:
        super().__init__()
        self.id = kwargs.get("id", None)
        self.model = model
        self.dataset = dataset

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def get_parameters(self):
        return self.model.get_parameters()
        
    def train(self, parameters):
        logger.log("Iniciando treinamento - Model {} - Dataset {}".format(self.model.uid, self.dataset.name))

        self.model.set_parameters(parameters)

        self.training_loop(self.dataset.train().dataloader())

        logger.log("Terminando treinamento - Model {} - Dataset {}".format(self.model.uid, self.dataset.name))

        self.model.checkpoint()

        return 

    def evaluate(self, parameters):

        logger.log("Iniciando validação - Model {} - Dataset {}".format(self.model.uid, self.dataset.name))
        
        self.model.set_parameters(parameters)
        
        self.validation_loop(self.dataset.validation().dataloader())

        logger.log("Terminando validação - Model {} - Dataset {}".format(self.model.uid, self.dataset.name))

        self.model.save()
        return

    def training_loop(self, data_loader):
        raise NotImplementedError("The training_loop method should be implemented!")

    def validation_loop(self, data_loader):
        raise NotImplementedError("The validation_loop method should be implemented!")
        
