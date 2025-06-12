
import uuid
from datetime import datetime
import os
import flautim2 as fl

from collections import OrderedDict

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, context, **kwargs):
        super(Model, self).__init__()
        
        self.uid = kwargs.get('id',str(uuid.uuid1()))
        self.name = kwargs.get('name', context.filesystem.user)
        self.suffix = kwargs.get('suffix', '')
        if self.suffix == '':
            self.suffix = 'FL-Global'
        self.version = kwargs.get('version', '1')
        
        self.path = context.filesystem.path
        #self.logger = kwargs.get('logger',None)

        self.file = "{}/models/{}{}.h5".format(self.path, self.name, self.suffix)
        self.checkpoint_file = "{}/models/{}{}-{}.h5"

    # def __log(self, msg, **kwargs):   # Dúvidas sobre essa função ???????????????
    #     if not self.logger is None:
    #         fl.log(msg, **kwargs)

    def set_parameters(self, parameters):
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)
        self.save()

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.state_dict().items()]
        
    def save(self):
        torch.save(self.state_dict(), self.file)

    def checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file.format(self.path, self.name, self.suffix, datetime.now()))
    
    def restore(self, file = None):
        if file is None:
            file = self.file
        if os.path.exists(file):
            self.load_state_dict(torch.load(file))

