from flautim2.pytorch.common import Backend, Logger, Measures, get_argparser, Config
import pandas as pd
import yaml
import argparse



def init():

    # with open("config.csv", "r") as file:
    #     ctx = yaml.safe_load(file)

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

    config_file = {
        "db": {
            'dbserver': ctx.dbserver,
            'dbport': ctx.dbport,
            'dbuser': ctx.dbuser,
            'dbpw': ctx.dbpw
        },
        "experiment": {
            "id": ctx.IDexperiment,
            'epochs': ctx.epochs,
            'rounds': ctx.rounds,
            'clients': ctx.clients
        },
        "filesystem": {
            'user': ctx.user,
            'path': ctx.path,
            'output_path': ctx.output_path,
        }
    }

    context = Config(config_file)

    context.backend = Backend(server = context.db.dbserver, port = context.db.dbport, user = context.db.dbuser, password = context.db.dbpw)
    context.logger = Logger(context.experiment.backend, context.filesystem.user)
    context.measures = Measures(context.experiment.backend, context.experiment.id)

    context.logger.log("Flautim2 initialized", context)
    
    return context

def log(message, context):
    # logger = Logger(ctx['backend'], ctx['context']['user'])
    context.logger.log(message, details="", object="", object_id = context.experiment.id)

    
    

def measures(experiment, metric, values, validation, context):
    # measures = Measures(ctx['backend'], ctx['context']['IDexperiment'])
    context.measures.log(experiment, metric, values, validation = False)

    
    


