import pymongo
from datetime import datetime
import argparse
from enum import Enum
import flwr as fl
import os, threading, schedule, logging
from typing import List, Tuple, Dict
import numpy as np
from pathlib import Path
import shutil
import time, traceback, subprocess, sys

from flwr.server.strategy.aggregate import weighted_loss_avg

class Backend(object):
    def __init__(self, **kwargs):
        super().__init__()
        self._server = kwargs.get('server', '127.0.0.1')
        self. _port = kwargs.get('port', '27017')
        self._user = kwargs.get('user', None)
        self._pw = kwargs.get('password', None)
        self._db = kwargs.get('authentication', 'admin')
        
    def get_db(self):
        self.connection = pymongo.MongoClient("mongodb://{}:{}@{}:{}".format(self._user, self._pw, self._server, self._port))
        self.db = self.connection["flautim"]
        return self.db
        
    def write_db(self, msg, collection):
        self.connection = pymongo.MongoClient("mongodb://{}:{}@{}:{}".format(self._user, self._pw, self._server, self._port))
        self.db = self.connection["flautim"]
            
        self.db[collection].insert_one(msg)
            
        self.connection.close()
        
    def write_experiment_results(self, file_path, experiment):
        db = self.get_db()["experiment_results"]

        filter = {"Experiment" : experiment}

        cursor = db.find(filter)

        with open(file_path, 'r') as file:
            content = file.read()

        doc = next(cursor, None)

        if doc is None:
            registro = {"Experiment": experiment, "content": content}
            db.insert_one(registro)
        else:
            newvalues = { "$set": { "content": content } }
            db.update_one(filter, newvalues)

    
    def write_experiment_results_callback(self, file_path, experiment):
        def fn_callback():
            self.write_experiment_results(file_path=file_path, experiment=experiment)
                
        return fn_callback
        


class Logger(object):
    def __init__(self, backend, context):
        super().__init__()
        
        self.backend = backend
        
        self.context = context
        
    def log(self, msg : str, **append):
        ts = str(datetime.now())
        data = { "user": self.context.user, "timestamp": ts, "message" : msg }
        if append is not None:
                data.update(append)
        self.backend.write_db(data, collection = 'logs')


class metrics(Enum):
     MSE = 1
     RMSE = 2
     NRMSE = 3
     MAE = 4
     MAPE = 5
     SMAPE = 6
     MDE = 7
     R2 = 8
     ACCURACY = 9
     PRECISION = 10
     RECALL = 11
     F1SCORE = 12
     AUC = 13
     CROSSENTROPY = 14
     TIME = 15,
     OTHER1 = 16,
     OTHER2 = 17,
     OTHER3 = 18
     

class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"
    ERROR = "error"


class ExperimentContext(object):
    def __init__(self, context, no_db=False):
        super().__init__()
        
        self.context = context

        self.id = self.context.IDexperiment

        backend = Backend(server = self.context.dbserver, port = self.context.dbport, user = self.context.dbuser, password=self.context.dbpw)

        self.experiments = backend.get_db()['experimento']
        
        experiment = self.experiments.find({"_id": self.id}).next()

        self.project = experiment["projectId"]

        self.model = experiment["modelId"]

        self.dataset = experiment["datasetId"]

        self.acronym = experiment["acronym"]

    def status(self, stat: ExperimentStatus):
        filter = { '_id': self.id }
        newvalues = { "$set": { 'status': str(stat) } }
        self.experiments.update_one(filter, newvalues)


class Measures(object):
    def __init__(self, backend, context):
        super().__init__()
        
        self.context = context
    
        self.backend = backend
        
    def log(self, experiment, metric, values, validation = False, **append):
        ts = str(datetime.now())
        data = { "Experiment": self.context.IDexperiment, "user": experiment.model.suffix, "timestamp": ts,
                 "metric" : str(metric), "model" : experiment.model.uid, "dataset": experiment.dataset.name, 
                "values": values, "validation": validation,
                "epoch" : experiment.epoch_fl}
        data.update(append)
        
        self.backend.write_db(data, collection = 'measures')

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
    }
    return config

    
def run_centralized(experiment, name_log = 'centralized.log', post_processing_fn = [], **kwargs):

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

    _, ctx, backend, logger, _ = get_argparser()
    experiment_id = ctx.IDexperiment
    path = ctx.path
    output_path = ctx.output_path
    epochs = ctx.epochs

    logger.log("Starting Centralized Training", details="", object="experiment_run", object_id=experiment_id )

    def schedule_file_logging():
        schedule.every(2).seconds.do(backend.write_experiment_results_callback('./centralized.log', experiment_id)) 
    
        while True:
            schedule.run_pending()
            time.sleep(1)

    thread_schedulling = threading.Thread(target=schedule_file_logging)
    thread_schedulling.daemon = True
    thread_schedulling.start()

    try:
        update_experiment_status(backend, experiment_id, "running")  

        experiment.fit()
    
        update_experiment_status(backend, experiment_id, "finished") 

        copy_model_wights(path, output_path, experiment_id, logger) 

        logger.log("Finishing Centralized Training", details="", object="experiment_run", object_id=experiment_id )
    except Exception as ex:
        update_experiment_status(backend, experiment_id, "error")  
        logger.log("Error during Centralized Training", details=str(ex), object="experiment_run", object_id=experiment_id )
        logger.log("Stacktrace of Error during Centralized Training", details=traceback.format_exc(), object="experiment_run", object_id=experiment_id )
        
    
    backend.write_experiment_results('./centralized.log', experiment_id)


class CustomFedAvg(fl.server.strategy.FedAvg):

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ) :
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics, server_round)
        elif server_round == 1:  # Only log this warning once
            logger.log("No evaluate_metrics_aggregation_fn provided")
            
        return loss_aggregated, metrics_aggregated

    
def run_federated(client_fn, eval_fn, name_log = 'flower.log', post_processing_fn = [], **kwargs):

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

    _, ctx, backend, logger, _ = get_argparser()
    experiment_id = ctx.IDexperiment
    path = ctx.path
    output_path = ctx.output_path
    num_clients = kwargs.get("num_clients", ctx.clients)
    num_rounds = kwargs.get("num_rounds", ctx.rounds)
    
    logger.log("Starting Flower Engine", details="", object="experiment_run", object_id=experiment_id )

    def schedule_file_logging():
        schedule.every(2).seconds.do(backend.write_experiment_results_callback('./flower.log', experiment_id)) 
    
        while True:
            schedule.run_pending()
            time.sleep(1)

    thread_schedulling = threading.Thread(target=schedule_file_logging)
    thread_schedulling.daemon = True
    thread_schedulling.start()

    try:

        strategy = CustomFedAvg(
            fraction_fit=0.1,  
            fraction_evaluate=0.1,  
            min_available_clients=num_clients,  
            evaluate_fn=eval_fn,
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=client_fn("FL-Global").weighted_average
        )

        update_experiment_status(backend, experiment_id, "running")  
    
        client_resources = {"num_cpus": 1, "num_gpus": 0.25} # Cada cliente com 10% de gpu
 
        h = fl.simulation.start_simulation(
            client_resources=client_resources,
            client_fn=client_fn, 
            num_clients=num_clients, 
            config=fl.server.ServerConfig(num_rounds=num_rounds),  
            strategy=strategy,  
        )

        update_experiment_status(backend, experiment_id, "finished") 

        copy_model_wights(path, output_path, experiment_id, logger) 

        logger.log("Stopping Flower Engine", details="", object="experiment_run", object_id=experiment_id )
    except Exception as ex:
        update_experiment_status(backend, experiment_id, "error")  
        logger.log("Error while running Flower", details=str(ex), object="experiment_run", object_id=experiment_id )
        logger.log("Stacktrace of Error while running Flower", details=traceback.format_exc(), object="experiment_run", object_id=experiment_id )
    
    backend.write_experiment_results('./flower.log', experiment_id)

def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--user", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--dbserver", type=str, required=False, default="127.0.0.1")
    parser.add_argument("--dbport", type=str, required=False, default="27017")
    parser.add_argument("--dbuser", type=str, required=True)
    parser.add_argument("--dbpw", type=str, required=True)
    parser.add_argument("--clients", type=int, required=False, default=3)
    parser.add_argument("--rounds", type=int, required=False, default=10)
    parser.add_argument("--epochs", type=int, required=False, default=10)
    parser.add_argument("--IDexperiment", type=str, required=True, default=0)
    ctx = parser.parse_args()
    
    backend = Backend(server = ctx.dbserver, port = ctx.dbport, user = ctx.dbuser, password=ctx.dbpw)
    
    logger = Logger(backend, ctx)
    measures = Measures(backend, ctx)
    
    return parser, ctx, backend, logger, measures


def update_experiment_status(backend, id, status):
    filter = { '_id': id }
    newvalues = { "$set": { 'status': status } }
    experiments = backend.get_db()['experimento']
    experiments.update_one(filter, newvalues)


def copy_model_wights(path, output_path, id, logger):
    try:
        p = Path(path+"models/").glob('**/*')
        files = [x for x in p if x.is_file()]

        for file in files:
            if "FL-Global" in str(file.stem):
                nf = Path(output_path + str(id) + "_weights" + file.suffix)
                if nf.exists():
                    nf.unlink()
                shutil.copy(file.resolve(), nf.resolve())
                logger.log("Model weights successfully copied", details=file.name, object="filesystem_file", object_id=id )
    except Exception as e:
        logger.log("Erro while copying model wights", details=str(e), object="filesystem_file", object_id=id )

    


