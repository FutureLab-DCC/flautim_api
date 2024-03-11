import argparse
import numpy as np

import os
import flwr as fl
from flwr.common import NDArrays, Scalar
from common import context, logger, backend

class Experiment(fl.client.NumPyClient):
    def __init__(self, model : Model, dataset : Dataset, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.dataset = dataset
        
    def train(self, parameters, config):
        self.set_parameters(parameters)

        optim = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        
        train(self.model, self.trainloader, optim, epochs=self.epoch)

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        """Evaluate the model sent by the server on this client's
        local validation set. Then return performance metrics."""

        self.set_parameters(parameters)
        loss, accuracy = test(
            self.model, self.valloader
        )  
        torch.save(self.model.state_dict(), self.savemodel)
        return float(loss), len(self.valloader), {"accuracy": accuracy}
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--savemodel", type=str, required=True)
    args = parser.parse_args()

    dir_path = os.path.dirname(args.savemodel)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    
    client_fn_callback = generate_client_fn(args)
    
    # now we can define the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,  
        fraction_evaluate=0.1,  
        min_available_clients=3,  
        evaluate_fn=get_evalulate_fn(),
    )  
   
    history = fl.simulation.start_simulation(
    client_fn=client_fn_callback, 
    num_clients=3, 
    config=fl.server.ServerConfig(num_rounds=10),  
    strategy=strategy,  
    )        