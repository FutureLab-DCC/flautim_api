import pymongo
from datetime import datetime
import argparse
from enum import Enum
import flwr
import os, threading, schedule, logging
from typing import List, Tuple, Dict
import numpy as np
from pathlib import Path
import shutil
import time, traceback, sys

from flwr.server.strategy.aggregate import weighted_loss_avg

from flwr.server import ServerApp

from flwr.client import ClientApp

import platform
import psutil
import subprocess

from flautim.pytorch.h5_store import save_event, read_events, merge_experiment_h5, default_merged_path, list_shards, delete_events

import json
import threading
from bson import ObjectId 
from contextlib import nullcontext
 
# Desabilita o file locking do HDF5 para evitar erros de "Unable to lock file (errno=11)" em execuções com múltiplos
# processos ou em sistemas de arquivos compartilhados (NFS). No Flautim cada processo escreve em um shard diferente,
# portanto não há escrita concorrente no mesmo arquivo.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

def get_pod_log_info() -> str:
    info = []

    # OS and Python info
    info.append(f"OS: {platform.system()} {platform.release()}")
    info.append(f"Python Version: {platform.python_version()}")
    info.append(f"Machine: {platform.machine()}")
    info.append(f"Hostname: {platform.node()}")

    # CPU info
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq()
    info.append(f"CPU Cores: {cpu_count}")
    if cpu_freq:
        info.append(f"CPU Frequency: {cpu_freq.current:.2f} MHz")

    # Memory info
    mem = psutil.virtual_memory()
    info.append(f"Memory: {mem.total / (1024**3):.2f} GB")

    # GPU info using PyTorch (offline-friendly)
    try:
        import torch

        info.append(f"PyTorch Version: {torch.__version__}")
        # Isto NÃO é driver, mas ajuda a diagnosticar build/compatibilidade:
        info.append(f"CUDA Build Version (torch.version.cuda): {torch.version.cuda}")

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            info.append(f"CUDA Available: Yes ({n} device(s))")

            for idx in range(n):
                name = torch.cuda.get_device_name(idx)
                props = torch.cuda.get_device_properties(idx)
                total_mem_gb = props.total_memory / (1024**3)

                # Alguns campos úteis do get_device_properties:
                # props.major / props.minor (compute capability)
                # props.multi_processor_count
                info.append(
                    f"GPU {idx}: {name}, "
                    f"VRAM: {total_mem_gb:.2f} GB, "
                    f"Compute: {props.major}.{props.minor}, "
                    f"SMs: {props.multi_processor_count}"
                )
        else:
            info.append("CUDA Available: No")
    except Exception as e:
        # Se torch não estiver instalado, ou ambiente sem CUDA, etc.
        info.append(f"GPU Info: Not available via PyTorch ({type(e).__name__})")

    # Format the info in a box
    lines = [f"│ {line}" for line in info]
    width = max(len(line) for line in lines)
    header = " MACHINE SETTINGS "
    box_top = "\n" + f"┌{'─' * (width)}┐"
    title_line = f"│{header.center(width)}│"
    box_bottom = f"└{'─' * (width)}┘"

    return "\n".join([box_top, title_line] + lines + [box_bottom])


class Backend(object):
    def __init__(self, **kwargs):
        super().__init__()
        self._server = kwargs.get('server', '127.0.0.1')
        self. _port = kwargs.get('port', '27017')
        self._user = kwargs.get('user', None)
        self._pw = kwargs.get('password', None)
        self._db = kwargs.get('authentication', 'admin')
        self._db_name = kwargs.get("db_name", "flautim") 
        self._h5_dir = kwargs.get("h5_dir", None) 
        self._experiment_id = kwargs.get("experiment_id", None) 

    @property
    def connection_string(self):
        return f"mongodb://{self._user}:{self._pw}@{self._server}:{self._port}"
        
    def get_db(self):
        if self._server == None:
            return None
        
        self.connection = pymongo.MongoClient("mongodb://{}:{}@{}:{}".format(self._user, self._pw, self._server, self._port))
        self.db = self.connection["flautim"]
        
        return self.db
        
    def write_db(self, msg, collection):
        if self._server != None:
            with pymongo.MongoClient(self.connection_string) as client:
                db = client[self._db_name]
                db[collection].insert_one(msg)
        
        save_event( base_dir=self._h5_dir, experiment_id=str(self._experiment_id), collection=str(collection), doc=msg )
            
        #print("[DB save]", collection, msg , sep="|")


    def close_db(self):
        self.connection.close()

    def write_experiment_results(self, file_path, experiment):
		# Read file content
        with open(file_path, "r") as file:
            content = file.read()
		   
        if self._server != None:
            with pymongo.MongoClient(self.connection_string) as client:
                db = client[self._db_name]
                collection = db["experiment_results"]
                filter_query = {"Experiment": experiment}
 
				# Check if document exists, update or insert
                if collection.find_one(filter_query) is None:
                   collection.insert_one({"Experiment": experiment, "content": content})
                else:
                   collection.update_one(filter_query, {"$set": {"content": content}})
          

        delete_events(  base_dir=self._h5_dir, experiment_id=str(self._experiment_id), collection="experiment_results", where={"Experiment": experiment} )   
        save_event( base_dir=self._h5_dir, experiment_id=str(self._experiment_id), collection=str("experiment_results"), doc={"Experiment": experiment, "content": content} )
        #print("[DB save]", "experiment_results", str(content) , sep="|")   
                        
    
    def write_experiment_results_callback(self, file_path, experiment):
        def fn_callback():
            self.write_experiment_results(file_path=file_path, experiment=experiment)
                
        return fn_callback
        

def get_config():
	import yaml
	with open('./configs/config.yaml') as f:
		cfg = yaml.safe_load(f)
		
		ctx = argparse.ArgumentParser()			  
		ctx.user = cfg['user']
		ctx.path = cfg['path']
		ctx.output_path = cfg['output_path']
		ctx.dbserver = cfg['db_server']
		ctx.dbport = cfg['db_port']
		ctx.dbuser = cfg['db_user']
		ctx.dbpw = cfg['db_pw'] 
		#ctx.clients = "3"
		#ctx.round = "10"
		#ctx.epochs = "10"
		ctx.IDexperiment = cfg['experiment_id']
		ctx.h5_dir = cfg['h5_dir']
		
		backend = Backend(server = ctx.dbserver, port = ctx.dbport, user = ctx.dbuser, password=ctx.dbpw,
                               h5_dir = ctx.h5_dir, experiment_id = ctx.IDexperiment)

		logger = Logger(backend, ctx)
		measures = Measures(backend, ctx)
			   
		return cfg, ctx, backend, logger, measures


class Logger(object):
    def __init__(self, backend, context):
        super().__init__()
        
        self.backend = backend
        
        self.user = context.user
        
    def log(self, msg, details="", object="", object_id=None, **append):
        ts = str(datetime.now())
        data = { "user": self.user, "timestamp": ts, "message": msg, 
                "details": details, "object": object, "object_id": object_id }
        if append:
            data.update(append)
        self.backend.write_db(data, collection='logs')

class Measures(object):
    def __init__(self, backend, IDexperiment):
        super().__init__()
        
        self.IDexperiment = IDexperiment
    
        self.backend = backend
        
    def log(self, experiment, metric, values, validation = False, epoch = None, **append):
        ts = str(datetime.now())
        data = { "Experiment": self.IDexperiment, "user": str(experiment.model.suffix), "timestamp": ts,
                 "metric" : 'metrics.' + str(metric), "model" : experiment.model.uid, "dataset": experiment.dataset.name, 
                "values": values, "validation": validation,
                "epoch" : experiment.epochs if epoch is None else epoch }
        data.update(append)
        
        self.backend.write_db(data, collection = 'measures')

class Output(object):
    """
    Interface de alto nível para salvar *outputs* no HDF5 de forma simples,
    semelhante a flautim.log() e flautim.measures().

    Exemplos de uso pelo usuário:
        flautim.output.image(img_bytes)
        flautim.output.array(weights)
        flautim.output.text("modelo final salvo")
        flautim.output.json({"acc": 0.95, "epoch": 10})

    Cada chamada:
      1) salva o conteúdo físico no HDF5 (/outputs/blobs ou /outputs/arrays)
      2) gera um identificador único (ref)
      3) opcionalmente registra um evento em /outputs/events com metadados
    """

    def __init__(self, backend, context):
        """
        backend  -> backend de banco (Mongo), não usado diretamente aqui,
                    mas mantido para simetria com Logger/Measures.
        context  -> contexto global do experimento (contém h5_dir e experiment.id)
        """
        self.backend = backend
        self.context = context

    # ============================================================
    # IMAGEM (PNG/JPEG/etc.)
    # ============================================================
    def image(self, content: bytes, name=None, meta=None):
        """
        Salva uma imagem no HDF5.

        Parâmetros:
        - content : bytes da imagem (ex: PNG gerado com matplotlib)
        - name    : nome lógico (ex: "confusion_epoch_5.png")
        - meta    : dicionário opcional com metadados (epoch, acc, data, etc.)

        Retorno:
        - ref (string): identificador único do arquivo salvo no HDF5
        """
        from flautim2.pytorch.h5_store import save_output, save_event

        # Salva fisicamente a imagem em /outputs/blobs/<ref>
        ref = save_output(
            base_dir=self.context.filesystem.h5_dir,
            experiment_id=self.context.experiment.id,
            kind="image",
            content=content,
            name=name,
            mime="image/png",
        )

        # Se houver metadados, cria um evento em /outputs/events
        # para facilitar busca/auditoria
        if meta:
            save_event(
                base_dir=self.context.filesystem.h5_dir,
                experiment_id=self.context.experiment.id,
                collection="outputs",
                doc={"ref": ref, **meta},
            )

        return ref

    # ============================================================
    # ARRAY NUMÉRICO (numpy / torch -> numpy)
    # ============================================================
    def array(self, arr, name=None, meta=None):
        """
        Salva um array (ex: pesos, embeddings, matrizes) no HDF5.

        Parâmetros:
        - arr  : numpy array
        - name : nome lógico (ex: "weights_epoch_10")
        - meta : metadados opcionais

        O array é salvo em:
            /outputs/arrays/<ref>
        """
        from flautim2.pytorch.h5_store import save_output, save_event

        ref = save_output(
            base_dir=self.context.filesystem.h5_dir,
            experiment_id=self.context.experiment.id,
            kind="array",
            content=arr,
            name=name,
        )

        if meta:
            save_event(
                base_dir=self.context.filesystem.h5_dir,
                experiment_id=self.context.experiment.id,
                collection="outputs",
                doc={"ref": ref, **meta},
            )

        return ref

    # ============================================================
    # TEXTO SIMPLES
    # ============================================================
    def text(self, text: str, name=None, meta=None):
        """
        Salva um texto simples no HDF5.

        Exemplos:
            - logs longos
            - resumo do experimento
            - configuração em texto

        O texto é salvo como blob (bytes UTF-8).
        """
        from flautim2.pytorch.h5_store import save_output, save_event

        ref = save_output(
            base_dir=self.context.filesystem.h5_dir,
            experiment_id=self.context.experiment.id,
            kind="text",
            content=text,
            name=name,
        )

        if meta:
            save_event(
                base_dir=self.context.filesystem.h5_dir,
                experiment_id=self.context.experiment.id,
                collection="outputs",
                doc={"ref": ref, **meta},
            )

        return ref

    # ============================================================
    # JSON (dict/list)
    # ============================================================
    def json(self, obj, name=None, meta=None):
        """
        Salva um objeto Python (dict ou list) como JSON no HDF5.

        Exemplos:
            flautim.output.json({"epoch": 5, "acc": 0.91})

        Internamente:
          - o objeto é serializado com json.dumps
          - salvo como blob (application/json)

        Parâmetros:
        - obj  : dict ou list
        - name : nome lógico do arquivo (ex: "metrics_epoch_5.json")
        - meta : metadados opcionais

        Retorno:
        - ref (string): identificador único no HDF5
        """
        from .h5_store import save_output, save_event
        import json

        # Serializa o objeto Python para string JSON
        payload = json.dumps(obj, ensure_ascii=False, indent=2)

        # Salva o JSON como blob no HDF5
        ref = save_output(
            base_dir=self.context.filesystem.h5_dir,
            experiment_id=self.context.experiment.id,
            kind="json",
            content=payload,                 # string JSON
            name=name or "data.json",
            mime="application/json",
        )

        # Registra evento de output para indexação/busca
        if meta:
            doc = {"ref": ref, "kind": "json", "name": name or "data.json", **meta}
            save_event(
                base_dir=self.context.filesystem.h5_dir,
                experiment_id=self.context.experiment.id,
                collection="outputs",
                doc=doc,
            )

        return ref

        
                  
class ExperimentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"
    ERROR = "error"

def get_experiment_variables(context, all_var = False):
    # backend = Backend(
    #     server = context.db.dbserver,
    #     port = context.db.dbport,
    #     user = context.db.dbuser,
    #     password = context.db.dbpw
    # )
    # Use context manager to avoid leaks
    if context.backend._server == None:
        experiment_variables = read_events( base_dir=context.backend._h5_dir, experiment_id=context.experiment.id, from_all_shards = True, collection = "experimento", where={"experiment_id": context.experiment.id } )[-1]
        if all_var == True:
            return experiment_variables
        else:
            return {"projectId": experiment_variables["experiment_id"],
				"modelId": experiment_variables["modelId"],
				"datasetId": experiment_variables["datasetId"],
				"acronym": experiment_variables["acronym"] }
        
        #return {"projectId": "",
        #    "modelId":  "",
        #    "datasetId":  "",
        #    "acronym":  "" }
    else:
        with pymongo.MongoClient(context.backend.connection_string) as client:
            db = client["flautim"]
            experiments = db["experimento"]
            experiment = experiments.find_one({"_id": context.experiment.id})
		   
            if all_var == True:
                 return experiment
            else:
                 return {"projectId": experiment["projectId"],
						"modelId": experiment["modelId"],
						"datasetId": experiment["datasetId"],
						"acronym": experiment["acronym"]}


class ExperimentContext(object):
    def __init__(self, context, no_db=False):
        super().__init__()

        variables = get_experiment_variables(context)

        # Assign fetched variables to class attributes
        self.project = variables["projectId"]
        self.model = variables["modelId"]
        self.dataset = variables["datasetId"]
        self.acronym = variables["acronym"]
        self.h5_dir = context.backend._h5_dir

    def status(self, stat: ExperimentStatus):   
        filter = { '_id': self.id }
        newvalues = { "$set": { 'status': str(stat) } }
        self.experiments.update_one(filter, newvalues)

        experiment_variables = read_events( base_dir=self.h5_dir, experiment_id=self.id , collection = "experimento", where={"experiment_id": self.id  } )[-1]
        experiment_variables["lastupdate"] = str(datetime.now())
        experiment_variables["status"] = stat 
        save_event( base_dir=self.h5_dir, experiment_id=self.id , collection="experimento", doc=experiment_variables ) 




def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
    }
    return config


class CustomFedAvg(flwr.server.strategy.FedAvg):

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ) :
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics, server_round)
        elif server_round == 1:  # Only log this warning once
            logger.log("No evaluate_metrics_aggregation_fn provided")
            
        return loss_aggregated, metrics_aggregated


def weighted_average(metrics) :
    """Compute weighted average.

    It is a generic implementation that averages only over floats and ints and drops the
    other data types of the Metrics.
    """
    # num_samples_list can represent the number of samples
    # or the number of batches depending on the client
    num_samples_list = [n_batches for n_batches, _ in metrics]
    num_samples_sum = sum(num_samples_list)
    metrics_lists: Dict[str, List[float]] = {}
    for num_samples, all_metrics_dict in metrics:
        #  Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            if isinstance(value, (float, int)):
                metrics_lists[single_metric] = []
        # Just one iteration needed to initialize the keywords
        break

    for num_samples, all_metrics_dict in metrics:
        # Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            # Add weighted metric
            if isinstance(value, (float, int)):
                metrics_lists[single_metric].append(float(num_samples * value))

    weighted_metrics: Dict[str, Scalar] = {}
    for metric_name, metric_values in metrics_lists.items():
        weighted_metrics[metric_name] = sum(metric_values) / num_samples_sum

    return weighted_metrics


def run_federated(client_fn, server_fn, name_log = 'flower.log', post_processing_fn = [], **kwargs):

    #self.metrics = Config(metrics) 
    logging.basicConfig(filename=name_log,
                    filemode='w',  # 'a' para append, 'w' para sobrescrever
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


    flower_logger = logging.getLogger('flwr')
    flower_logger.setLevel(logging.INFO)  # Ajustar conforme necessário


    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    flower_logger.addHandler(console_handler)

    #_, ctx, backend, logger, _ = get_argparser()
    _, ctx, backend, logger, _ = get_config()
    experiment_id = ctx.IDexperiment
    path = ctx.path
    output_path = ctx.output_path
    num_clients = kwargs.get("num_clients", 10)
    num_rounds = 15 #kwargs.get("num_rounds", ctx.rounds)

    experiment_variables = read_events( base_dir=backend._h5_dir, experiment_id=experiment_id, collection = "experimento", where={"experiment_id": experiment_id} )[-1]

    
    logger.log("Starting Flower Engine", details="", object="experiment_run", object_id=experiment_id )
    logger.log(get_pod_log_info(), details="", object="experiment_run", object_id=experiment_id )

    def schedule_file_logging():
        schedule.every(2).seconds.do(backend.write_experiment_results_callback('./flower.log', experiment_id)) 
    
        while True:
            schedule.run_pending()
            time.sleep(1)

    thread_schedulling = threading.Thread(target=schedule_file_logging)
    thread_schedulling.daemon = True
    thread_schedulling.start()

    #fraction_fit = kwargs.get('fraction_fit', 1.)
    #fraction_evaluate  = kwargs.get('fraction_evaluate', 1.)

    try:

        update_experiment_status(backend, experiment_id, "running")  
        
        # Duplica a entrada na coleção experimento em cada processo ajustando o client_fn do usuario
        _original_client_fn = client_fn
        #def client_fn(arg):  
        #    if len( read_events( base_dir=backend._h5_dir, experiment_id=experiment_id, collection = "experimento" ) ) == 0:
        #        save_event( base_dir=backend._h5_dir, experiment_id=experiment_id, collection="experimento", doc=experiment_variables )  
        #    return _original_client_fn(arg)
            
        client_app = ClientApp(client_fn=client_fn)
        server_app = ServerApp(server_fn=server_fn)

        #client_resources = kwargs.get('client_resources', {"num_cpus": 1, "num_gpus": 0.0})  
        try:
            import torch 
            if torch.cuda.is_available():
                client_resources = kwargs.get('client_resources', {"num_cpus": 1, "num_gpus": 0.5}) 
                logger.log("client_resources"+str(client_resources)) 
            else:
                client_resources = kwargs.get('client_resources', {"num_cpus": 1, "num_gpus": 0.0})
                logger.log("client_resources"+str(client_resources))
        except Exception as e: 
            client_resources = kwargs.get('client_resources', {"num_cpus": 1, "num_gpus": 0.0})
            logger.log("client_resources"+str(client_resources))
       
                
        flwr.simulation.run_simulation(server_app=server_app, client_app=client_app, 
                                     num_supernodes=num_clients,
                                     backend_config={"client_resources": client_resources})

        update_experiment_status(backend, experiment_id, "finished") 

        copy_model_wights(path, output_path, experiment_id, logger) 

        logger.log("Stopping Flower Engine", details="", object="experiment_run", object_id=experiment_id )


    except Exception as ex:
        update_experiment_status(backend, experiment_id, "error")  
        logger.log("Error while running Flower", details=str(ex), object="experiment_run", object_id=experiment_id )
        logger.log("Stacktrace of Error while running Flower", details=traceback.format_exc(), object="experiment_run", object_id=experiment_id )
    
    backend.write_experiment_results('./flower.log', experiment_id)


def update_experiment_status(backend, id, status):
	 
	experiment_variables = read_events( base_dir=backend._h5_dir, experiment_id=id, collection = "experimento", where={"experiment_id": id } )[-1]
	experiment_variables["lastupdate"] = str(datetime.now())
	experiment_variables["status"] = status
	save_event( base_dir=backend._h5_dir, experiment_id=id, collection="experimento", doc=experiment_variables )
		
	if backend._server != None:
		filter = { '_id': id }
		newvalues = { "$set": { 'status': status } }
		experiments = backend.get_db()['experimento']
		experiments.update_one(filter, newvalues)


def copy_model_wights(path, output_path, id, logger):
    try:
        p = Path(path+"models/").glob('**/*')
        files = [x for x in p if x.is_file()]

        for file in files:
            if "FL-Global" in str(file.stem):
                nf = Path(output_path + str(id) + "_weights" + file.suffix)
                if nf.exists():
                    nf.unlink()
                shutil.copy(file.resolve(), nf.resolve())
                logger.log("Model weights successfully copied", details=file.name, object="filesystem_file", object_id=id )
    except Exception as e:
        logger.log("Erro while copying model wights", details=str(e), object="filesystem_file", object_id=id )


class Config(dict):
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                value = Config(value)
            self[key] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Config object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value


def finalize_h5_merge(h5_dir, experiment_id):
    """
    Finaliza o armazenamento em HDF5 realizando o merge dos shards.

    - antes do merge, espera os shards ficarem 3s consecutivos sem alteração (size/mtime),
      para reduzir falhas na validação quando algum processo ainda está escrevendo.
    """ 

    # Verifica se já existe um arquivo merged
    merged_file = default_merged_path(h5_dir, experiment_id)

    if os.path.exists(merged_file):
        print(f"{datetime.now()} - Merge já realizado anteriormente: {merged_file}",file=sys.stderr,flush=True,)
        return

    # Espera 3s consecutivos sem alteração nos shards
    try:
        shard_files = list_shards(h5_dir, experiment_id)

        # Se não há shards, deixa o merge_experiment_h5 decidir (ele levanta FileNotFoundError)
        if shard_files:
            stable_required_s = 3.0
            poll_s = 0.5
            timeout_s = 60.0  # evita travar para sempre

            # Snapshot inicial
            last_stats = {}
            for f in shard_files:
                if os.path.exists(f):
                    st = os.stat(f)
                    last_stats[f] = (st.st_size, st.st_mtime)

            stable_start = time.time()
            wait_start = stable_start

            while True:
                time.sleep(poll_s)

                # Re-lista shards (se algum processo criou shard novo, conta como "mudou")
                current_shards = list_shards(h5_dir, experiment_id)
                if set(current_shards) != set(shard_files):
                    shard_files = current_shards
                    last_stats = {}
                    for f in shard_files:
                        if os.path.exists(f):
                            st = os.stat(f)
                            last_stats[f] = (st.st_size, st.st_mtime)
                    stable_start = time.time()

                changed = False

                for f in shard_files:
                    if not os.path.exists(f):
                        changed = True
                        last_stats.pop(f, None)
                        continue

                    st = os.stat(f)
                    cur = (st.st_size, st.st_mtime)
                    if last_stats.get(f) != cur:
                        changed = True
                        last_stats[f] = cur

                if changed:
                    stable_start = time.time()
                else:
                    if (time.time() - stable_start) >= stable_required_s:
                        break  # ficou 3s sem mudar

                if (time.time() - wait_start) >= timeout_s:
                    print(f"{datetime.now()} - Aviso: shards não estabilizaram em {timeout_s}s; seguindo com merge mesmo assim.",file=sys.stderr,flush=True,)
                    break

    except Exception as e:
        print(f"{datetime.now()} - Aviso: falha ao esperar estabilização dos shards: {str(e)}",file=sys.stderr,flush=True,)

    # Executa o merge dos shards
    try:
        print(f"{datetime.now()} - Fazendo merge dos H5 em: {h5_dir} (experiment_id={experiment_id})",file=sys.stderr,flush=True,)

        list_h5 = merge_experiment_h5(
            base_dir=h5_dir,
            experiment_id=experiment_id,
            delete_shards_on_success=True,
        )

        print(f"{datetime.now()} - Merge realizado com sucesso. Shards usados: {list_h5}",file=sys.stderr,flush=True,)

    except Exception as e:
        print(f"{datetime.now()} - Erro ao realizar merge dos H5: {str(e)}",file=sys.stderr,flush=True)
