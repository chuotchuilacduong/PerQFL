

import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.model_name= args.model_name
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_client_model(self):
        model_path = os.path.join("models", self.dataset, self.model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # Tạo tên file
        model_filename = f"{self.algorithm}_{self.model.__class__.__name__}_{self.dataset}_client_{self.id}.pt"
        model_path = os.path.join(model_path, model_filename)

        # --- THAY ĐỔI LOGIC KIỂM TRA ---
        # Ưu tiên lưu state_dict cho tất cả model để đảm bảo ổn định
        # Hoặc kiểm tra như phía server nếu bạn muốn phân biệt rõ ràng
        # is_quantum_model = isinstance(self.model, (HQCNN_Ang_noQP, HQCNN_CNN, Hybrid_QCNN, mimo_HQCNN_Ang_noQP)) # Cần import các lớp này

        # Cách đơn giản và an toàn nhất là luôn dùng state_dict
        try:
            #print(f"🔹 Lưu state_dict model client {self.id}: {model_filename}")
            torch.save(self.model.state_dict(), model_path)
        except Exception as e:
            print(f"⚠️ Lỗi khi lưu state_dict cho client {self.id}: {e}")
            # Có thể thử lưu cả model như một phương án dự phòng, nhưng ít khuyến khích
            # print(f"🔹 Thử lưu toàn bộ model client {self.id}: {model_filename}")
            # torch.save(self.model, model_path)


    def load_client_model(self):
        model_path = os.path.join("models", self.dataset, self.model_name)

        # Xác định tên file
        model_filename = f"{self.algorithm}_{self.model.__class__.__name__}_{self.dataset}_client_{self.id}.pt"
        model_path = os.path.join(model_path, model_filename)

        assert os.path.exists(model_path), f"⚠️ Model file client {model_filename} không tồn tại!"

        # --- THAY ĐỔI LOGIC KIỂM TRA ---
        # Ưu tiên tải state_dict
        try:
            #print(f"🔹 Tải model client {self.id}: {model_filename} từ state_dict().")
            # Cần tạo model cùng cấu trúc trước khi load state_dict
            # self.model đã được khởi tạo trong __init__ bằng deepcopy, nên cấu trúc đã đúng
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            #print(f"✅ Tải state_dict thành công cho client {self.id}.")
        except Exception as e_state_dict:
            print(f"⚠️ Không tải được state_dict cho client {self.id} ({e_state_dict}), thử tải toàn bộ model...")
            try:
                 loaded_model = torch.load(model_path, map_location=self.device)
                 # Kiểm tra xem có đúng lớp model không
                 if isinstance(loaded_model, type(self.model)):
                      self.model = loaded_model
                      #print(f"✅ Tải toàn bộ model thành công cho client {self.id}.")
                 else:
                     # Có thể file lưu là state_dict từ lần trước
                     #print(f"⚠️ File đã tải không phải là đối tượng model, thử load_state_dict...")
                     self.model.load_state_dict(loaded_model)
                     #print(f"✅ Tải state_dict thành công cho client {self.id} (sau khi thử tải toàn bộ).")
            except Exception as e_full_model:
                 print(f"❌ Lỗi nghiêm trọng: Không thể tải model client {self.id} {model_filename}")
                 raise e_full_model
    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
