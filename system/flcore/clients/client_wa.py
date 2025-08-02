# flcore/clients/client_wa.py

import torch
import numpy as np
import time
from flcore.clients.clientbase import Client

class clientWA(Client):
    """
    Client for Weighted Averaging personalization.
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # Tham số riêng cho Weighted Averaging
        self.alpha = args.alpha
        self.omega = args.omega

    def personalize_model(self, global_model):
        """
        Công thức: θ_p = (α * θ_avg + ω * θ_i) / (α + ω)
        """
        global_params = global_model.parameters()
        local_params = self.model.parameters()
        
        personalized_model_params = []
        for g_param, l_param in zip(global_params, local_params):
            p_param_data = (self.alpha * g_param.data + self.omega * l_param.data) / (self.alpha + self.omega)
            personalized_model_params.append(p_param_data)

        for old_param, new_param_data in zip(self.model.parameters(), personalized_model_params):
            old_param.data = new_param_data.clone()

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        
        start_time = time.time()
        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time