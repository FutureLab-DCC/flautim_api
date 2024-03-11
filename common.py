import pymongo
import time
import argparse

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


parser = argparse.ArgumentParser()
parser.add_argument("--user", type=str, required=True)
parser.add_argument("--path", type=str, required=True)
parser.add_argument("--dbserver", type=str, required=True)
parser.add_argument("--dbport", type=str, required=True)
parser.add_argument("--dbuser", type=str, required=True)
parser.add_argument("--dbpw", type=str, required=True)
context = parser.parse_args()

backend = Backend(server = context.dbserver, port = context.dbport, user = context.dbuser, password=context.dbpw)
logger = Logger(backend)

logger.log("Inicializando ambiente!")
