from flautim2.pytorch.common import Backend, Logger, Measures, Config
import pandas as pd
import yaml
import argparse

class Init:
    def __init__(self):
        self.context = None

_init_instance = None

def read_config():
        with open('./configs/config.yaml') as f:
            try: 
                cfg = yaml.safe_load(f)
                return cfg
            except Exception as ex:
                raise ex

def init():

    global _init_instance
    _init_instance = Init()

    config = read_config()

    config_file = {
        "db": {
            'dbserver': config['db_server'],
            'dbport': config['db_port'],
            'dbuser': config['db_user'],
            'dbpw': config['db_pw']
        },
        "experiment": {
            "id": config['experiment_id'],
            "name": config['experiment_name'],
            "file":config['experiment_file']
        },
        "filesystem": {
            'user': config['user'],
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


