from flautim2.pytorch.common import Backend, Logger, Measures
import yaml

_ctx = None

_backend = None

def init():

    print("Flautim inicializado!")

    with open("config.yaml", "r") as file:

        _ctx = yaml.safe_load(file)

    backend = Backend(server = _ctx.host, port = _ctx.port, user = _ctx.username, password= _ctx.password)
    return _ctx


def log():

    print("log!")

    Logger(_backend, _ctx)



def measures():

    print("measures!")

    Measures(_backend, _ctx)
