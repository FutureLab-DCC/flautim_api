from common import context, logger, backend
import uuid
from datetime import datetime

from collections import OrderedDict

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, **kwargs):
        super(Model, self).__init__()
        
        self.uid = kwargs.get('id',str(uuid.uuid1()))
        self.name = kwargs.get('name', context.user)
        self.version = kwargs.get('version', '1')

        self.file = "{}/models/{}.pt".format(context.path, self.name)
        self.checkpoint_file = "{}/models/{}-{}.pt"
        
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.state_dict().items()]
        
    def save(self):
        torch.save(self.state_dict(), self.file)
        logger.log('Model saved', model = self.uid)

    def checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file.format(context.path, self.name, datetime.now()))
        logger.log('Model checkpointed', model = self.uid)
    
    def restore(self, file = None):
        if file is None:
            file = self.file
        self.load_state_dict(torch.load(file))
        logger.log('Model restored', model = self.uid, file = file)
