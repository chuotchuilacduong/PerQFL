
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

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

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

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

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

        # Tạo tên file chứa thông tin thuật toán, model và dataset
        # Sử dụng self.model_name đã được gán trong __init__
        model_filename = f"{self.algorithm}_{self.model_name}_{self.dataset}_server.pt"
        model_path = os.path.join(model_path, model_filename)

        # --- THAY ĐỔI LOGIC KIỂM TRA ---
        # Kiểm tra xem model có phải là loại cần lưu state_dict hay không
        # (Bao gồm các HQCNN cũ và Hybrid_QCNN mới)
        # Cách 1: Kiểm tra sự tồn tại của các thuộc tính lượng tử cụ thể
        # has_quantum = hasattr(self.global_model, "quantum_layer") or hasattr(self.global_model, "q_layer")
        # Cách 2: Kiểm tra loại lớp (an toàn hơn nếu tên thuộc tính có thể thay đổi)
        is_quantum_model = isinstance(self.global_model, (HQCNN_Ang_noQP, HQCNN_CNN, Hybrid_QCNN, mimo_HQCNN_Ang_noQP)) # Thêm các lớp model lượng tử của bạn vào đây

        if is_quantum_model:
            print(f"🔹 Lưu model {model_filename} dưới dạng state_dict() do có thành phần lượng tử.")
            torch.save(self.global_model.state_dict(), model_path)
        else:
            # Lưu ý: Lưu toàn bộ model có thể không ổn định, cân nhắc luôn dùng state_dict
            print(f"🔹 Lưu toàn bộ model {model_filename} (Model không có thành phần lượng tử được nhận diện).")
            try:
                torch.save(self.global_model, model_path)
            except Exception as e:
                print(f"⚠️ Lỗi khi lưu toàn bộ model, thử lưu state_dict: {e}")
                torch.save(self.global_model.state_dict(), model_path)
                print(f"🔹 Đã lưu model {model_filename} dưới dạng state_dict() thay thế.")

    def load_model(self):
        model_path = os.path.join("models", self.dataset)

        # Xác định tên file theo định dạng đã lưu
        # Sử dụng self.model_name
        model_filename = f"{self.algorithm}_{self.model_name}_{self.dataset}_server.pt"
        model_path = os.path.join(model_path, model_filename)

        assert os.path.exists(model_path), f"⚠️ Model file {model_filename} không tồn tại!"

        # --- THAY ĐỔI LOGIC KIỂM TRA ---
        # Tương tự như khi lưu, kiểm tra xem có phải model lượng tử không
        is_quantum_model = isinstance(self.global_model, (HQCNN_Ang_noQP, HQCNN_CNN, Hybrid_QCNN, mimo_HQCNN_Ang_noQP)) # Cập nhật danh sách này nếu cần

        # Cũng kiểm tra xem file lưu là state_dict hay toàn bộ model
        # (Cách đơn giản là thử load state_dict trước)
        try:
            # Thử tải state_dict trước, cách này an toàn nhất
            print(f"🔹 Thử tải model {model_filename} từ state_dict().")
            state_dict = torch.load(model_path, map_location=self.device) # map_location để đảm bảo tải đúng device
            self.global_model.load_state_dict(state_dict)
            print(f"✅ Tải state_dict thành công cho model {model_filename}.")
        except Exception as e_state_dict:
            print(f"⚠️ Không tải được state_dict ({e_state_dict}), thử tải toàn bộ model...")
            try:
                 # Nếu không phải state_dict, thử tải toàn bộ model (ít khuyến khích hơn)
                loaded_model = torch.load(model_path, map_location=self.device)
                 # Cần đảm bảo cấu trúc lớp model khớp hoàn toàn
                if isinstance(loaded_model, type(self.global_model)):
                    self.global_model = loaded_model
                    print(f"✅ Tải toàn bộ model thành công cho {model_filename}.")
                else:
                    # Nếu load được nhưng không phải là model object (có thể là state_dict từ lần lưu trước đó)
                    print(f"⚠️ File đã tải không phải là đối tượng model đầy đủ, thử load_state_dict lại...")
                    self.global_model.load_state_dict(loaded_model) # Thử load như state_dict
                    print(f"✅ Tải state_dict thành công cho model {model_filename} (sau khi thử tải toàn bộ).")

            except Exception as e_full_model:
                print(f"❌ Lỗi nghiêm trọng: Không thể tải model {model_filename} bằng cả state_dict và load toàn bộ.")
                raise e_full_model # Hoặc xử lý lỗi khác
    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = f"{self.dataset}_{self.algorithm}_{self.model_name}"  # Thêm tên model vào
        result_path = "../results/"
        
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_name = "{}.h5".format(algo)
            file_path = os.path.join(result_path, file_name)
            print("File path: " + file_path)

            # Kiểm tra nếu tệp đã tồn tại và thêm số vào tên tệp nếu cần
            counter = 1
            while os.path.exists(file_path):
                file_name = "{}({}).h5".format(algo, counter)
                file_path = os.path.join(result_path, file_name)
                counter += 1

            # Lưu kết quả vào tệp HDF5
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
            
            print(f"Results saved to: {file_path}")

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

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

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

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

    # fine-tuning on new clients
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
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc
