import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from flcore.trainmodel.models import *

from utils.data_utils import read_client_data
from utils.dlg import DLG


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.model_name= args.model_name
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        # Global results
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.rs_test_f1 = []

        # Per-client results
        self.rs_clients_test_acc = []
        self.rs_clients_train_loss = []
        self.rs_clients_test_f1 = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args,
                            id=i,
                            train_samples=len(train_data),
                            test_samples=len(test_data),
                            train_slow=train_slow,
                            send_slow=send_slow)
            self.clients.append(client)

    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True
        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)
        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        self.global_model.load_state_dict(self.uploaded_models[0].state_dict())
        for param in self.global_model.parameters():
            param.data.zero_()
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model.load_state_dict(torch.load(model_path))

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    def save_round_weights(self, round_number):
        """
        Saves the global model weights for the current round.
        This function is designed to be overridden by subclasses (like FedPer) 
        for algorithm-specific saving logic.
        """
        round_save_path = os.path.join(self.save_folder_name, "models_storage", 
                                       self.dataset, self.algorithm, self.model_name, 
                                       f"round_{round_number}")
        os.makedirs(round_save_path, exist_ok=True)
        
        # Default behavior for algorithms like FedAvg: save the entire global model
        model_path = os.path.join(round_save_path, "global_model.pt")
        torch.save(self.global_model.state_dict(), model_path)
    
    def save_results(self):
        algo = f"{self.dataset}_{self.algorithm}_{self.model_name}"
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_name = "{}.h5".format(algo)
            file_path = os.path.join(result_path, file_name)
            print("File path: " + file_path)

            # ĐÂY LÀ ĐOẠN CODE ĐÚNG ĐỂ XỬ LÝ TRÙNG LẶP
            counter = 1
            while os.path.exists(file_path):
                file_name = "{}({}).h5".format(algo, counter)
                file_path = os.path.join(result_path, file_name)
                counter += 1

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_test_f1', data=self.rs_test_f1)
                hf.create_dataset('rs_clients_test_acc', data=self.rs_clients_test_acc)
                hf.create_dataset('rs_clients_train_loss', data=self.rs_clients_train_loss)
                hf.create_dataset('rs_clients_test_f1', data=self.rs_clients_test_f1)
            
            print(f"Results saved to: {file_path}")

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        tot_f1 = []
        for c in self.clients:
            ct, ns, auc, f1 = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            tot_f1.append(f1*ns)
            num_samples.append(ns)
        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc, tot_f1

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
        ids = [c.id for c in self.clients]
        return ids, num_samples, losses

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        
        ids, num_samples, tot_correct, tot_auc, tot_f1 = stats
        
        test_acc = sum(tot_correct) / sum(num_samples)
        test_auc = sum(tot_auc) / sum(num_samples)
        test_f1 = sum(tot_f1) / sum(num_samples)
        train_loss = sum(stats_train[2]) / sum(stats_train[1])
        
        accs = [a / n if n > 0 else 0 for a, n in zip(tot_correct, num_samples)]
        aucs = [a / n if n > 0 else 0 for a, n in zip(tot_auc, num_samples)]
        f1s = [f / n if n > 0 else 0 for f, n in zip(tot_f1, num_samples)]
        losses = [l / n if n > 0 else 0 for l, n in zip(stats_train[2], stats_train[1])]
        
        self.rs_test_acc.append(test_acc)
        self.rs_test_auc.append(test_auc)
        self.rs_test_f1.append(test_f1)
        self.rs_train_loss.append(train_loss)
        
        self.rs_clients_test_acc.append(accs)
        self.rs_clients_train_loss.append(losses)
        self.rs_clients_test_f1.append(f1s)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        #print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Averaged Test F1-score: {:.4f}".format(test_f1))
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        #print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        print("Std Test F1-score: {:.4f}".format(np.std(f1s)))

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        tot_f1 = [] # THÊM DÒNG NÀY
        for c in self.new_clients:
            # SỬA DÒNG NÀY: Nhận đủ 4 giá trị
            ct, ns, auc, f1 = c.test_metrics() 
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            tot_f1.append(f1*ns) # THÊM DÒNG NÀY
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        # SỬA DÒNG NÀY: Trả về F1-score
        return ids, num_samples, tot_correct, tot_auc, tot_f1