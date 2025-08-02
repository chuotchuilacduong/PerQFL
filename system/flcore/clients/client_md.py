# flcore/clients/client_md.py

import torch
import copy
import time
from flcore.clients.clientbase import Client

class clientMD(Client):
    """
    Client for Model Drift personalization.
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # Thuộc tính để lưu trạng thái model vòng trước
        self.previous_model_state = None

    def _calculate_euclidean_distance(self, model1_params, model2_params):
        dist = 0.0
        for p1, p2 in zip(model1_params, model2_params):
            dist += torch.norm(p1.data - p2.data)**2
        return torch.sqrt(dist)

    def personalize_model(self, global_model):
        
        if self.previous_model_state is None:
            self.set_parameters(global_model)
            return

        previous_model = copy.deepcopy(self.model)
        previous_model.load_state_dict(self.previous_model_state)
        
        local_params = list(self.model.parameters())
        global_params = list(global_model.parameters())
        previous_params = list(previous_model.parameters())
        
        ed_avg = self._calculate_euclidean_distance(local_params, global_params)
        ed_l = self._calculate_euclidean_distance(local_params, previous_params)

        if ed_avg < ed_l:
            self.set_parameters(global_model)
        else:
            avg_params = [(l.data + g.data) / 2.0 for l, g in zip(local_params, global_params)]
            for old_param, new_data in zip(self.model.parameters(), avg_params):
                old_param.data = new_data.clone()

    def train(self):
        # 1. Lưu trạng thái model trước khi huấn luyện
        self.previous_model_state = copy.deepcopy(self.model.state_dict())

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