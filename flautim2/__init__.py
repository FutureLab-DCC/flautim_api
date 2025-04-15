from flautim2.pytorch.common import Backend, Logger, Measures, get_argparser
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

    file = {
        'user': ctx.user,
        'path': ctx.path,
        'output_path': ctx.output_path,
        'dbserver': ctx.dbserver,
        'dbport': ctx.dbport,
        'dbuser': ctx.dbuser,
        'dbpw': ctx.dbpw,
        'clients': ctx.clients,
        'rounds': ctx.rounds,
        'epochs': ctx.epochs,
        'IDexperiment': ctx.IDexperiment
    }
    
    backend = Backend(server = file['dbserver'], port = file['dbport'], user = file['dbuser'], password = file['dbpw'])

    log(f"file: {file}", {
        'backend': backend,
        'context': file
    })


    return {
        'backend': backend,
        'context': file
    }

def log(message, ctx):
    logger = Logger(ctx['backend'], ctx['context']['user'])
    logger.log(message, details="", object="", object_id = ctx['context']['IDexperiment'])

    
    

def measures(experiment, metric, values, validation, ctx):
    measures = Measures(ctx['backend'], ctx['context']['IDexperiment'])
    measures.log(experiment, metric, values, validation = False)
