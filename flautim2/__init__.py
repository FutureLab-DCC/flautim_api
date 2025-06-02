from flautim2.pytorch.common import Backend, Logger, Measures, get_argparser, Config
import pandas as pd
import yaml
import argparse

class Init:
    def __init__(self):
        self.context = None

_init_instance = None

def read_config(name):
        with open('config.yaml') as f:
            try: 
                cfg = yaml.safe_load(f)[name]
                return cfg
            except Exception as ex:
                raise ex

def init():

    global _init_instance
    _init_instance = Init()

    config_file = read_config()

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


