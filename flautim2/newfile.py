import pymongo
from datetime import datetime
import argparse
from enum import Enum
import flwr as fl
import os, threading, schedule, logging
from typing import List, Tuple, Dict
import numpy as np
from pathlib import Path
import shutil
import time, traceback, subprocess, sys

from flwr.server.strategy.aggregate import weighted_loss_avg

def get_argparser():
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
    
    backend = Backend(server = ctx.dbserver, port = ctx.dbport, user = ctx.dbuser, password=ctx.dbpw)
    
    logger = Logger(backend, ctx)
    measures = Measures(backend, ctx)
    
    return parser, ctx, backend, logger, measures
