import argparse
from flautim2.pytorch import Dataset, common
import numpy as np
from enum import Enum
import os
import flwr as fl
import flautim2 as fl_log
from flautim2.pytorch import Model
from flautim2.pytorch.common import ExperimentContext, ExperimentStatus, generate_client_fn, evaluate_fn, generate_server_fn

from flautim2.pytorch.common import metrics

class Experiment(fl.client.NumPyClient):
    def __init__(self, model : Model, dataset : Dataset, context, **kwargs) -> None:
        super().__init__()
        self.id =  context.experiment.id
        self.model = model
        self.dataset = dataset
        
        self.epoch_fl = 0
        self.context = ExperimentContext(context)

        self.metrics = None

        self.log = context.logger.log
        self.measures = context.measures

        self.model.id = self.context.model
        self.dataset.id = self.context.dataset

        #self.model.logger = self.logger

    def status(self, stat: ExperimentStatus):
        try:
            self.context.status(stat)
        except Exception as ex:
            self.log("Error while updating status", details=str(ex), object="experiment_fit")

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def get_parameters(self, config):
        return self.model.get_parameters()
        
    def fit(self, parameters, config):
        return_dic = {}
        
        self.log(f"Model training started", details="", object="", object_id=self.id)

        self.model.set_parameters(parameters)

        self.epoch_fl = config["server_round"]

        values_metrics_train = self.training_loop(self.dataset.dataloader())

        self.log(f"Model training finished", details="", object="", object_id=self.id)

        for name in values_metrics_train:
                #self.log(f"Mesure: "+ 'metrics.' + str(name), details="", object="", object_id=self.id)
                self.measures.log(self, 'metrics.' + name, values_metrics_train[name], validation=False, epoch = self.epoch_fl)
                return_dic[name] = float(values_metrics_train[name])

        self.model.save()

        return self.model.get_parameters(), len(self.dataset.dataloader()), return_dic

    def evaluate(self, parameters, config):

        return_dic = {}
        
        self.log(f"Model evaluation started", details="", object="experiment_evaluate", object_id=self.id)
        
        self.model.set_parameters(parameters)

        self.epoch_fl = config["server_round"]
        
        loss, values_metrics_validation = self.validation_loop(self.dataset.dataloader(validation = True))

        self.log("Model training finished", details="", object="experiment_evaluate" )

        self.log(f"Mesure: "+ 'metrics.' + str(values_metrics_validation), details="", object="", object_id=self.id)

        for name in values_metrics_validation:
                #self.log(f"Mesure: "+ 'metrics.' + str(name) , details="", object="", object_id=self.id)
                self.measures.log(self, 'metrics.' + name, values_metrics_validation[name], validation=True, epoch = self.epoch_fl)
                return_dic[name] = float(values_metrics_validation[name])
        
        self.model.save()
        
        return float(loss), len(self.dataset.dataloader(validation = True)), return_dic

    def training_loop(self, data_loader):
        raise NotImplementedError("The training_loop method should be implemented!")

    def validation_loop(self, data_loader):
        raise NotImplementedError("The validation_loop method should be implemented!")


    def run_federated(self, Dataset, Model, context, files, strategy, num_rounds, metrics, num_clients = 4, name_log = 'flower.log', post_processing_fn = [], **kwargs):

        metrics['LOSS'] = None
        self.metrics = Config(metrics)
    
        client_fn = generate_client_fn(context, files, Model, Dataset, self)
        evaluate_fn_callback = evaluate_fn(context, files, Model, self, Dataset)
        server_fn = generate_server_fn(context, evaluate_fn_callback, Model, strategy, num_rounds)
    
        logging.basicConfig(filename=name_log,
                        filemode='w',  # 'a' para append, 'w' para sobrescrever
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    
    
        flower_logger = logging.getLogger('flwr')
        flower_logger.setLevel(logging.INFO)  # Ajustar conforme necessário
    
    
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        flower_logger.addHandler(console_handler)
    
        #_, ctx, backend, logger, _ = get_argparser()
        experiment_id = self.id
        path = self.context.filesystem.path
        output_path = self.context.filesystem.output_path
        
        fl_log.log("Starting Flower Engine", details="", object="experiment_run", object_id=experiment_id )
        fl_log.log(get_pod_log_info(), details="", object="experiment_run", object_id=experiment_id )
    
        fl_log.log("1 - " + str(self.metrics), details="", object="experiment_run", object_id=experiment_id )
        fl_log.log("2 - " + str(Config(metrics)), details="", object="experiment_run", object_id=experiment_id )
    
        def schedule_file_logging():
            schedule.every(2).seconds.do(self.context.backend.write_experiment_results_callback('./flower.log', experiment_id)) 
        
            while True:
                schedule.run_pending()
                time.sleep(1)
    
        thread_schedulling = threading.Thread(target=schedule_file_logging)
        thread_schedulling.daemon = True
        thread_schedulling.start()
    
        #fraction_fit = kwargs.get('fraction_fit', 1.)
        #fraction_evaluate  = kwargs.get('fraction_evaluate', 1.)
    
        try:
    
            update_experiment_status(self.context.backend, experiment_id, "running")  
            
            client_app = ClientApp(client_fn=client_fn)
            server_app = ServerApp(server_fn=server_fn)
            
            flwr.simulation.run_simulation(server_app=server_app, client_app=client_app, 
                                         num_supernodes=num_clients,
                                         backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.5}})
    
            update_experiment_status(self.context.backend, experiment_id, "finished") 
    
            copy_model_wights(path, output_path, experiment_id, self.context.logger) 
    
            fl_log.log("Stopping Flower Engine", details="", object="experiment_run", object_id=experiment_id )
        except Exception as ex:
            update_experiment_status(self.context.backend, experiment_id, "error")  
            fl_log.log("Error while running Flower", details=str(ex), object="experiment_run", object_id=experiment_id )
            fl_log.log("Stacktrace of Error while running Flower", details=traceback.format_exc(), object="experiment_run", object_id=experiment_id )
        
        self.context.backend.write_experiment_results('./flower.log', experiment_id)
        
