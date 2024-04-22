import argparse
import numpy as np

import os
import flwr as fl
#from flwr.common import NDArrays, Scalar
from programming_api import Model, Dataset

class Experiment(fl.client.NumPyClient):
    def __init__(self, model : Model, dataset : Dataset, measures, logger,context, **kwargs) -> None:
        super().__init__()
        self.id = context.IDexperiment
        self.model = model
        self.dataset = dataset
        
        self.measures = measures
        
        self.epoch_fl = 0
        
        self.logger = logger

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def get_parameters(self, config):
        return self.model.get_parameters()
        
    def fit(self, parameters, config):
        self.logger.log("Iniciando treinamento - Experiment {} - User {} - Model {} - Dataset {}".format(self.id, self.model.suffix, self.model.uid, self.dataset.name))

        self.model.set_parameters(parameters)
        
        self.epoch_fl = config["server_round"]

        loss, acc = self.training_loop(self.dataset.dataloader())

        self.logger.log("Terminando treinamento - Experiment {} - User {} - Model {} - Dataset {}".format(self.id, self.model.suffix, self.model.uid, self.dataset.name))

        self.model.save()
        

        return self.model.get_parameters(), len(self.dataset.dataloader()), {"accuracy": float(acc)}

    def evaluate(self, parameters, config):

        self.logger.log("Iniciando validação - Experiment {} - User {} - Model {} - Dataset {}".format(self.id, self.model.suffix, self.model.uid, self.dataset.name))
        
        self.model.set_parameters(parameters)
        
        loss, acc = self.validation_loop(self.dataset.dataloader(validation = True))

        self.logger.log("Terminando validação - Experiment {} - User {} - Model {} - Dataset {}".format(self.id, self.model.suffix, self.model.uid, self.dataset.name))
        

        self.model.save()
        
        return float(loss), len(self.dataset.dataloader()), {"accuracy": float(acc)}

    def training_loop(self, data_loader):
        raise NotImplementedError("The training_loop method should be implemented!")

    def validation_loop(self, data_loader):
        raise NotImplementedError("The validation_loop method should be implemented!")
        
