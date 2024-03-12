import pymongo
import time
import argparse
from enum import Enum
import flwr as fl
#from flwr.common import NDArrays, Scalar

class Backend(object):
    def __init__(self, **kwargs):
        super().__init__()
        _server = kwargs.get('server', '127.0.0.1')
        _port = kwargs.get('port', '27017')
        _user = kwargs.get('user', None)
        _pw = kwargs.get('password', None)
        _db = kwargs.get('authentication', 'admin')
        
        
        self.connection = pymongo.MongoClient("mongodb://{}:{}@{}:{}".format(_user, _pw, _server, _port))
        self.db = self.connection["futurelab"]
        

class Logger(object):
    def __init__(self, backend):
        super().__init__()
        
        self.backend = backend
        self.logs = self.backend.db['logs']
        
    def log(self, msg : str, **append):
        ts = time.time()
        data = { "user": context.user, "timestamp": ts, "message" : msg }
        if append is not None:
                data.update(append)
        self.logs.insert_one(data)


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
    def __init__(self, backend):
        super().__init__()
        
        self.backend = backend
        self.measures = self.backend.db['measures']
        
    def log(self, experiment, metric, values, validation = False, **append):
        ts = time.time()
        data = { "user": context.user, "timestamp": ts, "metric" : str(metric),
                "model" : experiment.model.uid, "dataset": experiment.dataset.name, 
                "values": values, "validation": validation}
        data.update(append)
        
        self.measures.insert_one(data)


def run(client_fn, eval_fn):
    
    logger.log("Iniciando experimento")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,  
        fraction_evaluate=0.1,  
        min_available_clients=context.clients,  
        evaluate_fn=eval_fn(),
    )  
   
    history = fl.simulation.start_simulation(
        client_fn=client_fn, 
        num_clients=context.clients, 
        config=fl.server.ServerConfig(num_rounds=context.rounds),  
        strategy=strategy,  
    )

    logger.log("Finalizando experimento")


parser = argparse.ArgumentParser()
parser.add_argument("--user", type=str, required=True)
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--dbserver", type=str, required=False, default="127.0.0.1")
parser.add_argument("--dbport", type=str, required=False, default="270017")
parser.add_argument("--dbuser", type=str, required=True)
parser.add_argument("--dbpw", type=str, required=True)
parser.add_argument("--clients", type=int, required=False, default=3)
parser.add_argument("--rounds", type=int, required=False, default=10)
context = parser.parse_args()

backend = Backend(server = context.dbserver, port = context.dbport, user = context.dbuser, password=context.dbpw)
logger = Logger(backend)
measures = Measures(backend)

# logger.log("Inicializando ambiente!")
