from flautim.pytorch.common import Backend, Logger, Measures, Config, Output, get_experiment_variables, finalize_h5_merge
from flautim.pytorch.h5_store import save_event, save_output, default_merged_path
import pandas as pd
import yaml
import argparse

import uuid
from datetime import datetime 
from functools import partial
import os
import atexit
import signal
import sys

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

def create_offline_config(experiment_name: str, path_run_experiment: str):
	if path_run_experiment is None:
		raise ValueError("The path_run_experiment parameter cannot be None.")
    
    # Verifica se já existe um arquivo de configuração; caso exista, não o sobrescreve e encerra a função.
	if os.path.isfile( str(os.path.join(path_run_experiment, "configs", "config.yaml"))  ):
		config = read_config() 
		experiment_id = config['experiment_id']
		print(f"A configuration file already exists with experiment id '{experiment_id}'. Flautim will run using the existing configuration file.") 
		return
	else:
		print(f"Configuration file does not exist. A new one will be created at '{path_run_experiment}/config/' .")           
	
	# Define o conteúdo padrão
	uu_id = "local-"+str(uuid.uuid4()) 
	default_config = {
		"db_server": None,
		"db_port" : None,
		"db_user" : None,
		"db_pw"   : None,
		"experiment_id"  : uu_id,
		"experiment_name": experiment_name,
		"experiment_file": "-",
		"user": "Local_PC",
		"path": path_run_experiment,
		"output_path": path_run_experiment,
		"h5_dir": path_run_experiment+"/outputs"
		}

	# Cria pastas ./configs ./models ./outputs
	configs_dir = os.path.join(path_run_experiment, "configs")
	os.makedirs(configs_dir, exist_ok=True)
		
	models_dir = os.path.join(path_run_experiment, "models")
	os.makedirs(models_dir, exist_ok=True)
	
	models_dir = os.path.join(path_run_experiment, "outputs")
	os.makedirs(models_dir, exist_ok=True)

	# Caminho do arquivo
	config_file = os.path.join(configs_dir, "config.yaml")

	# Escreve o YAML
	with open(config_file, "w") as f:
		yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)

	print(f"[OK] Configuration file created in: {config_file}")
    
    
def init(use_db_server = True):
    
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
            'h5_dir': config['h5_dir'],
        }
    }

    context = Config(config_file)

    # Caso já exista um arquivo HDF5 com o mesmo ID, pode ser que o usuário esteja reexecutando um experimento.
    # Nesse caso, o arquivo HDF5 criado anteriormente deve ser removido.
    merged_path_file = default_merged_path(context.filesystem.h5_dir, context.experiment.id) 
    if os.path.isfile(merged_path_file):
        print(f"Warning: An HDF5 file already exists for experiment with ID '{context.experiment.id}'. As a new execution is starting, the file will be deleted.")
        os.remove(merged_path_file)

 
    context.backend = Backend(server = context.db.dbserver, port = context.db.dbport,
                               user = context.db.dbuser, password = context.db.dbpw, 
                               h5_dir = context.filesystem.h5_dir, experiment_id = context.experiment.id)
                               
    context.logger = Logger(context.backend, context.filesystem)
    context.measures = Measures(context.backend, context.experiment.id)
    context.output = Output(context.backend, context)

    if context.db.dbserver == None: 
        experiment_variables = {
			"_id":context.experiment.id,
			"name":("Local_"+context.experiment.name), 
			"nameLowerCase":("local_"+context.experiment.name.lower()), 
			"information":"-",
			"hyperparameterFile":"-",
			"apiFile":"-",
			"datasetId":"-",
			"modelId":"-",
			"projectId":"-",
			"instituteId":"-",
			"acronym":"-",
			"status":"running",
			"createdby":os.environ.get('USER'),
			"createdat":str(datetime.now()), 
			"lastupdate":str(datetime.now()), 
			"updatedby":"-"
		}
    else:  
        experiment_variables = get_experiment_variables(context, True)
		
    save_event( base_dir=context.filesystem.h5_dir, experiment_id=context.experiment.id, collection="experimento", doc=experiment_variables ) 

    _init_instance.context = context
    
    # Tenta assegura que o merge dos shards HDF5 seja realizado durante o encerramento do programa.
    atexit.register(partial(finalize_h5_merge, context.filesystem.h5_dir, context.experiment.id))
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    sys.stdout = PrintLogger(
        sys.__stdout__,
        context.filesystem.h5_dir,
        context.experiment.id,
        "stdout"
    )

    sys.stderr = PrintLogger(
        sys.__stderr__,
        context.filesystem.h5_dir,
        context.experiment.id,
        "stderr"
    )

    log(f"h5_dir: {context.filesystem.h5_dir}") #TODO: remover depois
    return context
 
def finalize_experiment(): 
    finalize_h5_merge( _init_instance.context.filesystem.h5_dir, _init_instance.context.experiment.id )  

def _handle_signal(sig, frame):
    log(f"handle_signal: {sig}")
    finalize_h5_merge( _init_instance.context.filesystem.h5_dir, _init_instance.context.experiment.id )
    sys.exit(0)
    
def log(message, details = "", object = ""):
    _init_instance.context.logger.log(message, details=str(details), object=str(object), object_id=_init_instance.context.experiment.id)
    
def measures(experiment, metric, values, validation = False):
    experiment.context.measures.log(experiment, metric, values, validation)

def output_image(content: bytes, name=None, meta=None):
    return _init_instance.context.output.image(content, name=name, meta=meta)

def output_array(arr, name=None, meta=None):
    return _init_instance.context.output.array(arr, name=name, meta=meta)

def output_text(text: str, name=None, meta=None):
    return _init_instance.context.output.text(text, name=name, meta=meta)

def output_json(obj, name=None, meta=None):
    return _init_instance.context.output.json(obj, name=name, meta=meta)

class PrintLogger:
    def __init__(self, original_stream, h5_dir, experiment_id, stream_name="stdout"):
        self.original_stream = original_stream
        self.h5_dir = h5_dir
        self.experiment_id = experiment_id
        self.stream_name = stream_name
        self.buffer = ""
        self.saving = False

    def write(self, text):
        self.original_stream.write(text)
        self.buffer += text

        if self.saving:
            return

        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                try:
                    self.saving = True
                    save_event(
                        base_dir=self.h5_dir,
                        experiment_id=str(self.experiment_id),
                        collection="prints_log",
                        doc={
                            "message": line,
                            "stream": self.stream_name
                        }
                    )
                except Exception:
                    # nunca deixa quebrar stdout/stderr
                    pass
                finally:
                    self.saving = False

    def flush(self):
        self.original_stream.flush()

    def fileno(self):
        return self.original_stream.fileno()

    def isatty(self):
        return self.original_stream.isatty()