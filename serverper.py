import random
import time
import os
import torch
from flcore.clients.clientper import clientPer
from flcore.servers.serverbase import Server
from threading import Thread

class FedPer(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientPer)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            # Client training
            for client in self.selected_clients:
                client.train()

            self.receive_models()
            
            # Aggregate models if any were received
            if len(self.uploaded_models) > 0:
                self.aggregate_parameters()
                
                # Save weights after successful aggregation
                self.save_round_weights(i)

            self.Budget.append(time.time() - s_t)
            print('-'*25, f' Round {i} finished in {self.Budget[-1]:.2f}s ', '-'*25)

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientPer)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    # CORRECTED: Only the base model is received from clients
    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            # Simplified time cost check for clarity, your logic was fine
            tot_samples += client.train_samples
            # IMPORTANT: The client sends its updated base model
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.base)
        
        for i, w in enumerate(self.uploaded_weights):
            if tot_samples > 0:
                self.uploaded_weights[i] = w / tot_samples

    # CORRECTED: The server's global_model IS the base model
    def send_models(self):
        assert len(self.selected_clients) > 0
        for client in self.selected_clients:
            start_time = time.time()
            # self.global_model IS the base model.
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    # CORRECTED: Aggregation logic for the base model
    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        # Zero out the server's global (base) model parameters
        for param in self.global_model.parameters():
            param.data.zero_()
        
        # Aggregate the received base models
        for w, client_base_model in zip(self.uploaded_weights, self.uploaded_models):
            for server_param, client_param in zip(self.global_model.parameters(), client_base_model.parameters()):
                server_param.data += client_param.data.clone() * w

    # CORRECT: This function is perfectly implemented
    def save_round_weights(self, round_number):
        round_save_path = os.path.join(self.save_folder_name, "models_storage", 
                                       self.dataset, self.algorithm, self.model_name, 
                                       f"round_{round_number}")
        os.makedirs(round_save_path, exist_ok=True)

        # 1. Save the global model (which is the base part)
        global_model_path = os.path.join(round_save_path, "global_model.pt")
        torch.save(self.global_model.state_dict(), global_model_path)

        # 2. Save the local head of each participating client
        for client in self.selected_clients:
            client_head_path = os.path.join(round_save_path, f"client_{client.id}_head.pt")
            torch.save(client.model.head.state_dict(), client_head_path)
        
        print(f"Saved FedPer models for round {round_number} to: {round_save_path}")