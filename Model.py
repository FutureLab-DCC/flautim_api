
import uuid
from datetime import datetime
import os

from collections import OrderedDict

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, context, **kwargs):
        super(Model, self).__init__()
        
        self.uid = kwargs.get('id',str(uuid.uuid1()))
        self.name = kwargs.get('name', context.user)
        self.suffix = kwargs.get('suffix', '')
        self.version = kwargs.get('version', '1')
        
        self.path = context.path
        #self.logger = logger

        self.file = "{}/models/{}{}.h5".format(self.path, self.name, self.suffix)
        self.checkpoint_file = "{}/models/{}{}-{}.h5"

    def set_parameters(self, parameters):
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.state_dict().items()]
        
    def save(self):
        torch.save(self.state_dict(), self.file)
        #self.logger.log('Model saved', model = self.uid)

    def checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file.format(self.path, self.name, self.suffix, datetime.now()))
        #self.logger.log('Model checkpointed', model = self.uid)
    
    def restore(self, file = None):
        if file is None:
            file = self.file
        if os.path.exists(file):
            self.load_state_dict(torch.load(file))
            #self.logger.log('Model restored', model = self.uid, file = file)
