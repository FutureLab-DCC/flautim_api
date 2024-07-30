import argparse
import numpy as np

import os
import flwr as fl
#from flwr.common import NDArrays, Scalar
from flautim import Model, Dataset, common

class ExperimentContext(object):
    def __init__(self, context, no_db=False):
        super().__init__()
        
        self.context = context

        self.id = self.context.IDexperiment

        backend = common.Backend(server = self.context.dbserver, port = self.context.dbport, user = self.context.dbuser, password=self.context.dbpw)

        experiments = backend.get_db()['experimento']
        
        experiment = experiments.find({"_id": self.id}).next()

        self.project = experiment["projectId"]

        self.model = experiment["modelId"]

        self.dataset = experiment["datasetId"]

        self.acronym = experiment["acronym"]


class Experiment(fl.client.NumPyClient):
    def __init__(self, model : Model, dataset : Dataset, measures, logger, context, **kwargs) -> None:
        super().__init__()
        self.id = context.IDexperiment
        self.model = model
        self.dataset = dataset
        
        self.measures = measures
        
        self.epoch_fl = 0
        
        self.logger = logger

        self.context = ExperimentContext(context)

        self.model.id = self.context.model
        self.dataset.id = self.context.dataset

        self.model.logger = self.logger

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def get_parameters(self, config):
        return self.model.get_parameters()
        
    def fit(self, parameters, config):
        self.logger.log("Iniciando treinamento do modelo", details="", object="experiment_fit", object_id=self.id )

        self.model.set_parameters(parameters)
        
        self.epoch_fl = config["server_round"]

        loss, acc = self.training_loop(self.dataset.dataloader())

        self.logger.log("Finalizando treinamento do modelo", details="", object="experiment_fit", object_id=self.id )

        self.model.save()

        return self.model.get_parameters(), len(self.dataset.dataloader()), {"accuracy": float(acc)}

    def evaluate(self, parameters, config):

        self.logger.log("Iniciando avaliação do modelo", details="", object="experiment_evaluate", object_id=self.id )
        
        self.model.set_parameters(parameters)
        
        loss, acc = self.validation_loop(self.dataset.dataloader(validation = True))

        self.logger.log("Finalizando avaliação do modelo", details="", object="experiment_evaluate", object_id=self.id )
        
        self.model.save()
        
        return float(loss), len(self.dataset.dataloader()), {"accuracy": float(acc), "loss" : float(loss)}

    def training_loop(self, data_loader):
        raise NotImplementedError("The training_loop method should be implemented!")

    def validation_loop(self, data_loader):
        raise NotImplementedError("The validation_loop method should be implemented!")
        
