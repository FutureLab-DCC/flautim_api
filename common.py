import pymongo
import time
import argparse
from enum import Enum
import flwr as fl
#from flwr.common import NDArrays, Scalar

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
    def __init__(self, backend, context):
        super().__init__()
        
        self.backend = backend
        self.context = context
        
    def log(self, experiment, metric, values, validation = False, **append):
        ts = time.time()
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

def run(client_fn, eval_fn):

    parser, context, backend, logger, measures = get_argparser()
    
    logger.log("Iniciando experimento")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,  
        fraction_evaluate=0.1,  
        min_available_clients=context.clients,  
        evaluate_fn=eval_fn,
        on_fit_config_fn=fit_config
    )  
   
    history = fl.simulation.start_simulation(
        client_fn=client_fn, 
        num_clients=context.clients, 
        config=fl.server.ServerConfig(num_rounds=context.rounds),  
        strategy=strategy,  
    )

    logger.log("Finalizando experimento")

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
    measures = Measures(backend, context)
    
    return parser, context, backend, logger, measures

# logger.log("Inicializando ambiente!")
