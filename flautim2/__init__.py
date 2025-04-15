from flautim2.pytorch.common import Backend, Logger, Measures, get_argparser
import pandas as pd
import yaml



def init():

    parser, context, backend, logger, measures = get_argparser()


    logger.log("Olá! Esta é a função init!", details="", object="init", object_id=context.IDexperiment)

    # with open("config.csv", "r") as file:
    #     _ctx = yaml.safe_load(file)

    #_ctx['experiment']['id'] = context.IDexperiment

    # logger.log(f"ID: {_ctx['experiment']['id']}", details="", object="init", object_id=context.IDexperiment)


    # _backend = Backend(server = _ctx['db']['host'], port = _ctx['db']['port'], user = _ctx['db']['username'], password= _ctx['db']['password'])
    # _logger = logger(_backend, _ctx)
    # _measures = Measures(_backend, _ctx)

    ctx = {
        'backend': backend,
        'context': context,
        'logger': logger,
        'measures': measures
    }

    return ctx

def log(message, ctx):
    
    ctx['logger'].log(message, details="", object="init", object_id=ctx['context'].IDexperiment)

    
    

def measures(experiment, metric, values, validation, ctx):
    
    ctx['measures'].log(experiment, metric, values, validation = False)
