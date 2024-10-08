import argparse
from flautim.pytorch import Dataset, common
from enum import Enum
import os
from flautim.pytorch import Model
from flautim.pytorch.common import ExperimentContext, ExperimentStatus
import time

class Experiment(object):
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
        self.epochs = kwargs.get('epochs', 1)
        self.epoch_fl = 0

    def status(self, stat: ExperimentStatus):
        try:
            self.context.status(stat)
        except Exception as ex:
            self.logger.log("Error while updating status", details=str(ex), object="experiment_fit", object_id=self.id )

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def get_parameters(self, config):
        return self.model.get_parameters()
        
    def fit(self, **kwargs):
        self.logger.log("Model training started", details="", object="experiment_fit", object_id=self.id )

        for epoca in range(self.epochs):
            start_time = time.time()
            epoch_loss, acc = self.training_loop(self.dataset.dataloader())
            elapsed_time = time.time() - start_time
            self.logger.log(f'[TRAIN] Epoch [{epoca+1}] Training Loss: {epoch_loss:.4f}, ' +
                f'Time: {elapsed_time:.2f} seconds', details="", object="experiment_fit", object_id=self.id )

        self.logger.log("Model training finished", details="", object="experiment_fit", object_id=self.id )

        self.model.save()

    def training_loop(self, data_loader):
        raise NotImplementedError("The training_loop method should be implemented!")
