import argparse
from flautim2.pytorch import Dataset, common
import numpy as np
from enum import Enum
import os
import flwr as fl
import flautim2 as fl_log
from flautim2.pytorch import Model
from flautim2.pytorch.common import ExperimentContext, ExperimentStatus

from flautim2.pytorch.common import metrics

class Experiment(fl.client.NumPyClient):
    def __init__(self, model : Model, dataset : Dataset, context, **kwargs) -> None:
        super().__init__()
        self.id =  context.experiment.id
        self.model = model
        self.dataset = dataset
        
        self.epoch_fl = 0
        self.context = ExperimentContext(context)

        self.log = context.logger.log
        self.measures = context.measures.log

        self.model.id = self.context.model
        self.dataset.id = self.context.dataset

        #self.model.logger = self.logger

    def status(self, stat: ExperimentStatus):
        try:
            self.context.status(stat)
        except Exception as ex:
            fl_log.log("Error while updating status", details=str(ex), object="experiment_fit")

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def get_parameters(self, config):
        return self.model.get_parameters()
        
    def fit(self, parameters, config):
        return_dic = {}
        
        self.log(f"Model training started", details="", object="", object_id=self.id)

        self.model.set_parameters(parameters)

        #self.epoch_fl = config["server_round"]

        values_metrics_train = self.training_loop(self.dataset.dataloader())

        self.log(f"Model training finished", details="", object="", object_id=self.id)

        for name in values_metrics_train:
                self.log(f"Mesure: "+ 'metrics.' + str(name), details="", object="", object_id=self.id)
                self.measures.log(self, 'metrics.' + name, values_metrics_train[name], validation=False)
                return_dic[name] = float(values_metrics_train[name])

        self.model.save()

        return self.model.get_parameters(), len(self.dataset.dataloader()), return_dic

    def evaluate(self, parameters, config):

        return_dic = {}
        
        self.log(f"Model evaluation started", details="", object="experiment_evaluate", object_id=self.id)
        
        self.model.set_parameters(parameters)
        
        values_metrics_validation = self.validation_loop(self.dataset.dataloader(validation = True))

        self.log("Model training finished", details="", object="experiment_evaluate" )

        self.log(f"Mesure: "+ 'metrics.' + str(values_metrics_validation), details="", object="", object_id=self.id)

        for name in values_metrics_validation:
                self.log(f"Mesure: "+ 'metrics.' + str(name) , details="", object="", object_id=self.id)
                self.measures.log(self, 'metrics.' + name, values_metrics_validation[name], validation=True)
                return_dic[name] = float(values_metrics_validation[name])
        
        self.model.save()
        
        return float(loss), len(self.dataset.dataloader(validation = True)), return_dic

    def training_loop(self, data_loader):
        raise NotImplementedError("The training_loop method should be implemented!")

    def validation_loop(self, data_loader):
        raise NotImplementedError("The validation_loop method should be implemented!")
    
