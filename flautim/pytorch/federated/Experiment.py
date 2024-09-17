import argparse
from flautim.pytorch import Dataset, common
import numpy as np
from enum import Enum
import os
import flwr as fl
from flautim.pytorch import Model
from flautim.pytorch.common import ExperimentContext, ExperimentStatus

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

    def status(self, stat: ExperimentStatus):
        try:
            self.context.status(stat)
        except Exception as ex:
            self.logger.log("Error while updating status", details=str(ex), object="experiment_fit", object_id=self.id )

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def get_parameters(self, config):
        return self.model.get_parameters()
        
    def fit(self, parameters, config):
        self.logger.log("Model training started", details="", object="experiment_fit", object_id=self.id )

        self.model.set_parameters(parameters)
        
        self.epoch_fl = config["server_round"]

        loss, acc = self.training_loop(self.dataset.dataloader())

        self.logger.log("Model training finished", details="", object="experiment_fit", object_id=self.id )

        self.model.save()

        return self.model.get_parameters(), len(self.dataset.dataloader()), {"accuracy": float(acc)}

    def evaluate(self, parameters, config):

        self.logger.log("Model evaluation started", details="", object="experiment_evaluate", object_id=self.id )
        
        self.model.set_parameters(parameters)
        
        loss, acc = self.validation_loop(self.dataset.dataloader(validation = True))

        self.logger.log("Model training finished", details="", object="experiment_evaluate", object_id=self.id )
        
        self.model.save()
        
        return float(loss), len(self.dataset.dataloader()), {"accuracy": float(acc), "loss" : float(loss)}

    def training_loop(self, data_loader):
        raise NotImplementedError("The training_loop method should be implemented!")

    def validation_loop(self, data_loader):
        raise NotImplementedError("The validation_loop method should be implemented!")
    

class CentralizedExperiment(object):
    def __init__(self, model : Model, dataset : Dataset, measures, logger, context, **kwargs) -> None:
        super().__init__()
        self.id = context.IDexperiment
        self.model = model
        self.dataset = dataset
        
        self.measures = measures
        
        self.logger = logger

        self.context = ExperimentContext(context)

        self.model.id = self.context.model
        self.dataset.id = self.context.dataset

        self.model.logger = self.logger

    def status(self, stat: ExperimentStatus):
        try:
            self.context.status(stat)
        except Exception as ex:
            self.logger.log("Error while updating status", details=str(ex), object="experiment_fit", object_id=self.id )

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def get_parameters(self, config):
        return self.model.get_parameters()
        
    def fit(self, parameters, config):
        self.logger.log("Model training started", details="", object="experiment_fit", object_id=self.id )

        self.model.set_parameters(parameters)
        
        loss, acc = self.training_loop(self.dataset.dataloader())

        self.logger.log("Model training finished", details="", object="experiment_fit", object_id=self.id )

        self.model.save()

        return self.model.get_parameters(), len(self.dataset.dataloader()), {"accuracy": float(acc)}

    def evaluate(self, parameters, config):

        self.logger.log("Model evaluation started", details="", object="experiment_evaluate", object_id=self.id )
        
        self.model.set_parameters(parameters)
        
        loss, acc = self.validation_loop(self.dataset.dataloader(validation = True))

        self.logger.log("Model training finished", details="", object="experiment_evaluate", object_id=self.id )
        
        self.model.save()
        
        return float(loss), len(self.dataset.dataloader()), {"accuracy": float(acc), "loss" : float(loss)}

    def training_loop(self, data_loader):
        raise NotImplementedError("The training_loop method should be implemented!")

    def validation_loop(self, data_loader):
        raise NotImplementedError("The validation_loop method should be implemented!")
        
