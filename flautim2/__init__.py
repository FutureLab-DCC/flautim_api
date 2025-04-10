from flautim2.pytorch.common import Backend, Logger, Measures
import yaml

_ctx = None

_backend = None

def init():

    #print("Flautim inicializado!")

    with open("config.csv", "r") as file:

        _ctx = yaml.safe_load(file)
    

    backend = Backend(server = _ctx['mongodb']['host'], port = _ctx['mongodb']['port'], user = _ctx['mongodb']['username'], password= _ctx._ctx['mongodb']['password'])

    logger = Logger(_backend, _ctx)

    logger.log(f'[TRAIN] Epoch [{epoca}] Training Loss: {epoch_loss:.4f}, ' +
                f'Time: {elapsed_time:.2f} seconds', details="", object="experiment_fit", object_id=_ctx['experiment']['id'])
    
    return _ctx


def log():

    #print("log!")

    Logger(_backend, _ctx)



def measures():

    #print("measures!")

    Measures(_backend, _ctx)
