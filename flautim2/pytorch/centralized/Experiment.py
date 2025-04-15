import argparse
from flautim2.pytorch import Dataset, common
from enum import Enum
import os, threading, schedule, logging
import flautim2 as fl
from flautim2.pytorch import Model
from flautim2.pytorch.common import ExperimentContext, ExperimentStatus, metrics
import time

class Experiment(object):
    def __init__(self, model : Model, dataset : Dataset, context, **kwargs) -> None:
        super().__init__()
        self.id = context['context']['IDexperiment']
        self.model = model
        self.dataset = dataset
        self.backend = context['backend']

        self.context = ExperimentContext(context['context'])

        # self.model.id = self.context.model
        # self.dataset.id = self.context.dataset

        # self.model.logger = self.logger

        self.epochs = kwargs.get('epochs', 1)

        #self.epoch_fl = 0

    def status(self, stat: ExperimentStatus):
        try:
            self.context['status'](stat)
        except Exception as ex:
            #self.logger.log("Error while updating status", details=str(ex), object="experiment_fit", object_id=self.id )
            fl.log("Error while updating status", details=str(ex), object="experiment_fit", object_id=self.id )

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def get_parameters(self, config):
        return self.model.get_parameters()
        
    def fit(self, **kwargs):
        fl.log("Model training started", context=self.context)

        for epochs in range(1, self.epochs+1):
            start_time = time.time()
            epoch_loss, acc = self.training_loop(self.dataset.dataloader())
            elapsed_time = time.time() - start_time
            self.epochs = epochs
            # self.logger.log(f'[TRAIN] Epoch [{epoca}] Training Loss: {epoch_loss:.4f}, ' +
                # f'Time: {elapsed_time:.2f} seconds', details="", object="experiment_fit", object_id=self.id )
            
            fl.log(f'[TRAIN] Epoch [{epochs}] Training Loss: {epoch_loss:.4f}, ' +
                f'Time: {elapsed_time:.2f} seconds', context=self.context)
            
            #self.measures.log(self, metrics.CROSSENTROPY, epoch_loss, validation=False)
            #self.measures.log(self, metrics.ACCURACY, acc, validation=False)

        fl.log("Model training finished", context=self.context)

        self.model.save()

    def training_loop(self, data_loader):
        raise NotImplementedError("The training_loop method should be implemented!")
    
    def run(self, metrics, name_log = 'centralized.log', post_processing_fn = [], **kwargs):

        fl.log("Olá, estamos no run!", self.context.context)

        logging.basicConfig(filename=name_log,
                        filemode='w',  # 'a' para append, 'w' para sobrescrever
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
        
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        root.addHandler(console_handler)

        # _, ctx, backend, logger, _ = get_argparser()
        # ctx = fl.init()
        
        #experiment_id = self.context['IDexperiment']
        #path = self.context['path']
        #output_path = self.context['output_path']
        #epochs = self.context['epochs']

        fl.log("Starting Centralized Training", self.context.context)

        def schedule_file_logging():
            schedule.every(2).seconds.do(self.backend.write_experiment_results_callback('./centralized.log', self.id)) 
        
            while True:
                schedule.run_pending()
                time.sleep(1)

        thread_schedulling = threading.Thread(target=schedule_file_logging)
        thread_schedulling.daemon = True
        thread_schedulling.start()

        try:
            update_experiment_status(self.backend, self.id, "running")  

            self.fit()
        
            update_experiment_status(self.backend, self.id, "finished") 

            copy_model_wights(self.context.context['path'], self.context.context['output_path'], self.id, self.logger) 

            fl.log("Finishing Centralized Training", context=self.context.context)
        except Exception as ex:
            update_experiment_status(self.backend, self.id, "error")  
            fl.log("Error during Centralized Training", details=str(ex), context=self.context.context)
            fl.log("Stacktrace of Error during Centralized Training", details=traceback.format_exc(), context=self.context.context)
            
        
        self.backend.write_experiment_results('./centralized.log', self.id)
