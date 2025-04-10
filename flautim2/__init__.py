from flautim2.pytorch.common import Backend, Logger, Measures, get_argparser
import yaml

_ctx = None

_backend = None

def init():

    parser, context, backend, logger, measures = get_argparser()

    with open("config.csv", "r") as file:
        _ctx = yaml.safe_load(file)

    _ctx['experiment']['id'] = context.IDexperiment
    print(_ctx)

    backend = Backend(server = _ctx['mongodb']['host'], port = _ctx['mongodb']['port'], user = _ctx['mongodb']['username'], password= _ctx._ctx['mongodb']['password'])
   
    return _ctx


def log():

    #print("log!")

    Logger(_backend, _ctx)



def measures():

    #print("measures!")

    Measures(_backend, _ctx)
