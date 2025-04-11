from flautim2.pytorch.common import Backend, Logger, Measures, get_argparser
import yaml

_ctx = None

_backend = None

def init():

    parser, context, backend, logger, measures = get_argparser()

    logger = Logger(backend, context)
    logger.log("Olá! Esta é a função init!", details="", object="init", object_id=context.IDexperiment)
   
    return _ctx


def log():

    #print("log!")

    Logger(_backend, _ctx)



def measures():

    #print("measures!")

    Measures(_backend, _ctx)
