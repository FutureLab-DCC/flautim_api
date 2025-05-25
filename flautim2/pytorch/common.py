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

from flwr.server import ServerApp

from flwr.client import ClientApp

class Backend(object):
    def __init__(self, **kwargs):
        super().__init__()
        self._server = kwargs.get('server', '127.0.0.1')
        self. _port = kwargs.get('port', '27017')
        self._user = kwargs.get('user', None)
        self._pw = kwargs.get('password', None)
        self._db = kwargs.get('authentication', 'admin')
        self._db_name = kwargs.get("db_name", "flautim")

    @property
    def connection_string(self):
        return f"mongodb://{self._user}:{self._pw}@{self._server}:{self._port}"
        
    def get_db(self):
        self.connection = pymongo.MongoClient("mongodb://{}:{}@{}:{}".format(self._user, self._pw, self._server, self._port))
        self.db = self.connection["flautim"]
        return self.db
        
    def write_db(self, msg, collection):
        with pymongo.MongoClient(self.connection_string) as client:
            db = client[self._db_name]
            db[collection].insert_one(msg)

    def close_db(self):
        self.connection.close()
        
    def write_experiment_results(self, file_path, experiment):
        with pymongo.MongoClient(self.connection_string) as client:
            db = client[self._db_name]
            collection = db["experiment_results"]
            filter_query = {"Experiment": experiment}

            # Read file content
            with open(file_path, "r") as file:
                content = file.read()

            # Check if document exists, update or insert
            if collection.find_one(filter_query) is None:
                collection.insert_one({"Experiment": experiment, "content": content})
            else:
                collection.update_one(filter_query, {"$set": {"content": content}})

    
    def write_experiment_results_callback(self, file_path, experiment):
        def fn_callback():
            self.write_experiment_results(file_path=file_path, experiment=experiment)
                
        return fn_callback
        


class Logger(object):
    def __init__(self, backend, context):
        super().__init__()
        
        self.backend = backend
        
        self.user = context.user
        
    def log(self, msg, details="", object="", object_id=None, **append):
        ts = str(datetime.now())
        data = { "user": self.user, "timestamp": ts, "message": msg, 
                "details": details, "object": object, "object_id": object_id }
        if append:
            data.update(append)
        self.backend.write_db(data, collection='logs')

class Measures(object):
    def __init__(self, backend, IDexperiment):
        super().__init__()
        
        self.IDexperiment = IDexperiment
    
        self.backend = backend
        
    def log(self, experiment, metric, values, validation = False, epoch = None, **append):
        ts = str(datetime.now())
        data = { "Experiment": self.IDexperiment, "user": experiment.model.suffix, "timestamp": ts,
                 "metric" : str(metric), "model" : experiment.model.uid, "dataset": experiment.dataset.name, 
                "values": values, "validation": validation,
                "epoch" : experiment.epochs if epoch is None else epoch }
        data.update(append)
        
        self.backend.write_db(data, collection = 'measures')


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

def get_experiment_variables(context):
    # backend = Backend(
    #     server = context.db.dbserver,
    #     port = context.db.dbport,
    #     user = context.db.dbuser,
    #     password = context.db.dbpw
    # )
    # Use context manager to avoid leaks
    with pymongo.MongoClient(context.backend.connection_string) as client:
        db = client["flautim"]
        experiments = db["experimento"]
        experiment = experiments.find_one({"_id": context.experiment.id})
       
        return {"projectId": experiment["projectId"],
                "modelId": experiment["modelId"],
                "datasetId": experiment["datasetId"],
                "acronym": experiment["acronym"]}


class ExperimentContext(object):
    def __init__(self, context, no_db=False):
        super().__init__()

        variables = get_experiment_variables(context)

        # Assign fetched variables to class attributes
        self.project = variables["projectId"]
        self.model = variables["modelId"]
        self.dataset = variables["datasetId"]
        self.acronym = variables["acronym"]

    def status(self, stat: ExperimentStatus):
        filter = { '_id': self.id }
        newvalues = { "$set": { 'status': str(stat) } }
        self.experiments.update_one(filter, newvalues)




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


def weighted_average(metrics) :
    """Compute weighted average.

    It is a generic implementation that averages only over floats and ints and drops the
    other data types of the Metrics.
    """
    # num_samples_list can represent the number of samples
    # or the number of batches depending on the client
    num_samples_list = [n_batches for n_batches, _ in metrics]
    num_samples_sum = sum(num_samples_list)
    metrics_lists: Dict[str, List[float]] = {}
    for num_samples, all_metrics_dict in metrics:
        #  Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            if isinstance(value, (float, int)):
                metrics_lists[single_metric] = []
        # Just one iteration needed to initialize the keywords
        break

    for num_samples, all_metrics_dict in metrics:
        # Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            # Add weighted metric
            if isinstance(value, (float, int)):
                metrics_lists[single_metric].append(float(num_samples * value))

    weighted_metrics: Dict[str, Scalar] = {}
    for metric_name, metric_values in metrics_lists.items():
        weighted_metrics[metric_name] = sum(metric_values) / num_samples_sum

    return weighted_metrics


def run_federated(client_fn, server_fn, name_log = 'flower.log', post_processing_fn = [], **kwargs):

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

    fraction_fit = kwargs.get('fraction_fit', 1.)
    fraction_evaluate  = kwargs.get('fraction_evaluate', 1.)

    try:

        update_experiment_status(backend, experiment_id, "running")  
    
        client_resources = {"num_cpus": 1, "num_gpus": 0.75} 

        client_app = ClientApp(client_fn=client_fn)
        server_app = ServerApp(server_fn=server_fn)
        
        fl.simulation.run_simulation(server_app=server_app, client_app=client_app, num_supernodes=num_clients)

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


class Config(dict):
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                value = Config(value)
            self[key] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Config object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value



