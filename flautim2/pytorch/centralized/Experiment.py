from flautim2.pytorch import Dataset, common
from enum import Enum
import threading, schedule, logging
import flautim2 as fl
from flautim2.pytorch import Model
from flautim2.pytorch.common import ExperimentContext, ExperimentStatus, update_experiment_status, copy_model_wights, Config, metrics
import time

class Experiment(object):
    def __init__(self, model : Model, dataset : Dataset, context, **kwargs) -> None:
        super().__init__()
        self.id = context.experiment.id
        self.model = model
        self.dataset = dataset
        
        self.context = context
        self.experiment_context = ExperimentContext(context)

        self.metrics = None

        self.model.id = self.experiment_context.model
        self.dataset.id = self.experiment_context.dataset

        self.epochs = kwargs.get('epochs', 1)

    def status(self, stat: ExperimentStatus):
        try:
            self.experiment_context['status'](stat)
        except Exception as ex:
            fl.log(f"Error while updating status: {str(ex)}")

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def get_parameters(self, config):
        return self.model.get_parameters()
        
    def fit(self, **kwargs):

        fl.log(f"Model training started")

        for epochs in range(1, self.epochs+1):
            start_time = time.time()
            values_metrics_train = self.training_loop(self.dataset.dataloader())
            elapsed_time = time.time() - start_time
            self.epochs = epochs
            
            fl.log(f'[TRAIN] Epoch [{epochs}] Training Loss: {values_metrics_train['LOSS']:.4f}, ' +
                f'Time: {elapsed_time:.2f} seconds')
            
            for name in values_metrics_train:
                fl.measures(self, name, values_metrics_train[name], validation=False)

            self.evaluate()

        fl.log("Model training finished")

        
    
    def evaluate(self, **kwargs):

        start_time = time.time()
        values_metrics_validation = self.validation_loop(self.dataset.dataloader(validation = True))
        elapsed_time = time.time() - start_time

        fl.log(f'[TRAIN] Epoch [{self.epochs}] Test Loss: {values_metrics_train['LOSS']:.4f}, ' +
                f'Time: {elapsed_time:.2f} seconds')

        for name in values_metrics_validation:
                fl.measures(self, name, values_metrics_validation[name], validation=True)

        #fl.log("Model evaluate finished")


    def training_loop(self, data_loader):
        raise NotImplementedError("The training_loop method should be implemented!")
    
    
    def validation_loop(self, data_loader):
        raise NotImplementedError("The training_loop method should be implemented!")
    
    
    def run(self, metrics, name_log = 'centralized.log', post_processing_fn = [], **kwargs):

        metrics['LOSS'] = None

        self.metrics = Config(metrics)

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

        fl.log(common.get_pod_log_info())

        fl.log(f"Starting Centralized Training")

        def schedule_file_logging():
            schedule.every(2).seconds.do(self.context.backend.write_experiment_results_callback('./centralized.log', self.id)) 
        
            while True:
                schedule.run_pending()
                time.sleep(1)

        thread_schedulling = threading.Thread(target=schedule_file_logging)
        thread_schedulling.daemon = True
        thread_schedulling.start()


        try:
            update_experiment_status(self.context.backend, self.id, "running")  

            self.fit()

            #self.evaluate()
        
            update_experiment_status(self.context.backend, self.id, "finished")

            copy_model_wights(self.context.filesystem.path, self.context.filesystem.output_path, self.id, self.context.logger) 

            fl.log(f"Finishing Centralized Training")

        except Exception as ex:
            update_experiment_status(self.context.backend, self.id, "error")  
            fl.log(f"Error during Centralized Training: {str(ex)}")
            fl.log(f"Stacktrace of Error during Centralized Training: {traceback.format_exc()}")
            
        
        self.context.backend.write_experiment_results('./centralized.log', self.id)





