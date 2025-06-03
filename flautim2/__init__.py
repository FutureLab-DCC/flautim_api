from flautim2.pytorch.common import Backend, Logger, Measures, Config
import pandas as pd
import yaml
import argparse

class Init:
    def __init__(self):
        self.context = None

_init_instance = None

def read_config():
        with open('/mnt/sigla/ZGacoXKh/config.yaml') as f:
            try: 
                cfg = yaml.safe_load(f)
                return cfg
            except Exception as ex:
                raise ex

def init():

    global _init_instance
    _init_instance = Init()

    config = read_config()

    # config_file = {
    #     "db": read_config("db"),
    #     "experiment": read_config("experiment"),
    #     "filesystem": read_config("filesystem")
    # }

    # parser = argparse.ArgumentParser()
    
    # parser.add_argument("--user", type=str, required=True)
    # parser.add_argument("--path", type=str, required=True)
    # parser.add_argument("--output-path", type=str, required=True)
    # parser.add_argument("--dbserver", type=str, required=False, default="127.0.0.1")
    # parser.add_argument("--dbport", type=str, required=False, default="27017")
    # parser.add_argument("--dbuser", type=str, required=True)
    # parser.add_argument("--dbpw", type=str, required=True)
    # parser.add_argument("--clients", type=int, required=False, default=3)
    # parser.add_argument("--rounds", type=int, required=False, default=10)
    # parser.add_argument("--epochs", type=int, required=False, default=10)
    # parser.add_argument("--IDexperiment", type=str, required=True, default=0)
    # ctx = parser.parse_args()

    config_file = {
        "db": {
            'dbserver': config['db_server'],
            'dbport': config['db_port'],
            'dbuser': config['db_user'],
            'dbpw': config['db_pw']
        },
        "experiment": {
            "id": config['experiment_id'],
        },
        "filesystem": {
            'path': config['path'],
            'output_path': config['output_path'],
        }
    }

    context = Config(config_file)

    context.backend = Backend(server = context.db.dbserver, port = context.db.dbport,
                               user = context.db.dbuser, password = context.db.dbpw)
    context.logger = Logger(context.backend, context.filesystem)
    context.measures = Measures(context.backend, context.experiment.id)

    _init_instance.context = context

    return context
    

def log(message, details = "", object = ""):
    _init_instance.context.logger.log(message, details=str(details), object=str(object), object_id=_init_instance.context.experiment.id)
    
def measures(experiment, metric, values, validation = False):
    experiment.context.measures.log(experiment, metric, values, validation)


