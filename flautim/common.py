import pymongo
import time
import argparse
from enum import Enum
import flwr as fl
import os
#from flwr.common import NDArrays, Scalar

from typing import List, Tuple, Dict
import numpy as np

from flwr.server.strategy.aggregate import weighted_loss_avg

import logging

class Backend(object):
    def __init__(self, **kwargs):
        super().__init__()
        self._server = kwargs.get('server', '127.0.0.1')
        self. _port = kwargs.get('port', '27017')
        self._user = kwargs.get('user', None)
        self._pw = kwargs.get('password', None)
        self._db = kwargs.get('authentication', 'admin')
        
        
    def write_db(self, msg, collection):
        self.connection = pymongo.MongoClient("mongodb://{}:{}@{}:{}".format(self._user, self._pw, self._server, self._port))
        self.db = self.connection["admin"]
            
        self.db[collection].insert_one(msg)
            
        self.connection.close()
        
    def write_experiment_results(self, file_path, experiment):
        with open(file_path, 'r') as file:
            content = file.read()
        
        
        msg = {"Experiment": experiment,
               "content": content}
        self.write_db(msg, "experiment_results")
            
            
        

class Logger(object):
    def __init__(self, backend, context):
        super().__init__()
        
        self.backend = backend
        
        self.context = context
        
    def log(self, msg : str, **append):
        ts = time.time()
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
     TIME = 15


class Measures(object):
    def __init__(self, context):
        super().__init__()
        
        self.context = context
        
    def log(self, experiment, metric, values, validation = False, **append):
        ts = time.time()
        data = { "Experiment": self.context.IDexperiment, "user": experiment.model.suffix, "timestamp": ts,
                 "metric" : str(metric), "model" : experiment.model.uid, "dataset": experiment.dataset.name, 
                "values": values, "validation": validation,
                "epoch" : experiment.epoch_fl}
        data.update(append)
        
        backend = Backend(server = self.context.dbserver, port = self.context.dbport, user = self.context.dbuser, password=self.context.dbpw)
        
        backend.write_db(data, collection = 'measures')

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
    }
    return config


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
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
            
        return loss_aggregated, metrics_aggregated

    
def run(client_fn, eval_fn, name_log = 'flower.log'):

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

    parser, context, backend, logger, measures = get_argparser()
    
    logger.log("Iniciando experimento")

    strategy = CustomFedAvg(
        fraction_fit=0.1,  
        fraction_evaluate=0.1,  
        min_available_clients=context.clients,  
        evaluate_fn=eval_fn,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=client_fn(0).weighted_average
    )  
   
    history = fl.simulation.start_simulation(
        client_fn=client_fn, 
        num_clients=context.clients, 
        config=fl.server.ServerConfig(num_rounds=context.rounds),  
        strategy=strategy,  
    )

    logger.log("Finalizando experimento")
    
    backend.write_experiment_results('./flower.log', context.IDexperiment)

def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--user", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--dbserver", type=str, required=False, default="127.0.0.1")
    parser.add_argument("--dbport", type=str, required=False, default="27017")
    parser.add_argument("--dbuser", type=str, required=True)
    parser.add_argument("--dbpw", type=str, required=True)
    parser.add_argument("--clients", type=int, required=False, default=3)
    parser.add_argument("--rounds", type=int, required=False, default=10)
    parser.add_argument("--IDexperiment", type=str, required=True, default=0)
    context = parser.parse_args()
    
    backend = Backend(server = context.dbserver, port = context.dbport, user = context.dbuser, password=context.dbpw)
    
    logger = Logger(backend, context)
    measures = Measures(context)
    
    return parser, context, backend, logger, measures

# logger.log("Inicializando ambiente!")
