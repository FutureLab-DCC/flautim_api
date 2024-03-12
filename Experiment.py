import argparse
import numpy as np

import os
import flwr as fl
from flwr.common import NDArrays, Scalar
from common import context, logger, backend, measures
import Model, Dataset

class Experiment(fl.client.NumPyClient):
    def __init__(self, model : Model, dataset : Dataset, **kwargs) -> None:
        super().__init__()
        self.id = kwargs.get("id", None)
        self.model = model
        self.dataset = dataset

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def get_parameters(self, config):
        return self.model.get_parameters()
        
    def train(self, parameters, config):
        logger.log("Iniciando treinamento - Model {} - Dataset {}".format(self.model.uid, self.dataset.name))

        self.model.set_parameters(parameters)

        self.training_loop(self.dataset.train())

        logger.log("Terminando treinamento - Model {} - Dataset {}".format(self.model.uid, self.dataset.name))

        self.model.checkpoint()

        return 

    def evaluate(self, parameters, config):

        logger.log("Iniciando validação - Model {} - Dataset {}".format(self.model.uid, self.dataset.name))
        
        self.model.set_parameters(parameters)
        
        self.validation_loop(self.dataset.validation())

        logger.log("Terminando validação - Model {} - Dataset {}".format(self.model.uid, self.dataset.name))

        self.model.save()
        return

    def training_loop(self, data_loader):
        raise NotImplementedError("The training_loop method should be implemented!")

    def validation_loop(self, data_loader):
        raise NotImplementedError("The validation_loop method should be implemented!")
        
