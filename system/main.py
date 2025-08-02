import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import sys

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverper import FedPer
from utils.result_utils import average_data
from utils.mem_utils import MemReporter
from flcore.trainmodel.models import *
from flcore.servers.server_wa import FedAvgWA
from flcore.servers.server_md import FedAvgMD

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):
    time_list = []
    reporter = MemReporter()
    model_str = args.model
    args.model_name = str(copy.deepcopy(args.model))
    mimo = args.massive_mimo
    M = args.num_clients
    K = args.num_terminals
    tau_p = args.pilot_num
    # if mimo== "True":
    # env = Environment(M=args.num_clients, K=args.num_terminals, tau=args.pilot_num)  # Initialize Environment with num_clients as M
    # BETAA = env.compute_large_scale_fading()  # Generate data using Environment
    for i in range(args.prev, args.times):
        print(f"\n======= Running time: {i} =======")
        print("Creating server and clients")
        start = time.time()
        if model_str == "CNN":
            if mimo == "True":
                args.model = CNNModel(M, K, tau_p).to(args.device)
            else:
                if "MNIST" in args.dataset or "EMNIST" in args.dataset or "FashionMNIST" in args.dataset:
                    args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes).to(args.device)
                elif "Cifar10" in args.dataset or "Cifar100" in args.dataset:
                    args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes).to(args.device)
                elif "TinyImagenet" in args.dataset:
                    return FedAvgCNN(in_features=3, num_classes=args.num_classes).to(args.device)
                else:
                    args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes).to(args.device)
        elif model_str == "MLP":
            if mimo == "True":
                args.model = MLPModel(M, K, tau_p, n_qubits).to(args.device)
            else:
                if "MNIST" in args.dataset or "EMNIST" in args.dataset or "FashionMNIST" in args.dataset:
                    args.model = FedAvgMLP(in_features=784, num_classes=args.num_classes).to(args.device)
                elif "Cifar10" in args.dataset or "Cifar100" in args.dataset:
                    args.model = FedAvgMLP(in_features=3072, num_classes=args.num_classes).to(args.device)
                elif "TinyImagenet" in args.dataset:
                    args.model = FedAvgMLP(in_features=12288, num_classes=args.num_classes).to(args.device)
        elif model_str == "HQCNN":
            weight_shapes = {
                "weights_0": 3,
                "weights_1": 3,
                "weights_2": 1,
                "weights_3": 1,
                "weights_4": 1,
                "weights_5": 3,
                "weights_6": 3,
            }
            if mimo == "True":
                args.model = HQCNN_Ang_noQP(M, K, tau_p, n_qubits).to(args.device)
            else:
                if "MNIST" in args.dataset:
                    args.model = HQCNN_Ang_noQP(in_features=28 * 28, num_classes=args.num_classes,
                                                weight_shapes=weight_shapes).to(args.device)
                elif "Cifar10" in args.dataset:
                    args.model = HQCNN_Ang_noQP(in_features=3 * 32 * 32, num_classes=args.num_classes,
                                                weight_shapes=weight_shapes).to(args.device)
        # Trong main.py, bên trong hàm run(args)
        elif model_str == "Quanv": #Quanvolution
            if mimo == "True":
                raise NotImplementedError("MIMO version for this specific Hybrid_QCNN not implemented yet.")
            else:
                if mimo == "True":
                    raise NotImplementedError("MIMO version for this specific Hybrid_QCNN not implemented yet.")
                else:
                    # Định nghĩa weight_shapes ở đây (giống nhau cho cả MNIST và Cifar10)
                    hybrid_weight_shapes = {
                        # "strong_layer_weights": (1, 4, 3),
                        # # Assuming n_layers=1, n_qubits_hybrid=4 for StronglyEntanglingLayers
                        "weights_0": 3,  # Shape for the 1st parameter of the first U_SU4
                        "weights_1": 3,  # Shape for the 2nd parameter of the first U_SU4
                        "weights_2": 1,  # Shape for the 3rd parameter of the first U_SU4
                        "weights_3": 1,  # Shape for the 4th parameter of the first U_SU4
                        "weights_4": 1,  # Shape for the 5th parameter of the first U_SU4
                        "weights_5": 3,  # Shape for the 6th parameter of the first U_SU4
                        "weights_6": 3,  # Shape for the 7th parameter of the first U_SU4
                        # Weights for the second U_SU4 layer
                        "weights_7": 3,  # Shape for the 1st parameter of the second U_SU4
                        "weights_8": 3,  # Shape for the 2nd parameter of the second U_SU4
                        "weights_9": 1,  # Shape for the 3rd parameter of the second U_SU4
                        "weights_10": 1,  # Shape for the 4th parameter of the second U_SU4
                        "weights_11": 1,  # Shape for the 5th parameter of the second U_SU4
                        "weights_12": 3,  # Shape for the 6th parameter of the second U_SU4
                        "weights_13": 3  # Shape for the 7th parameter of the second U_SU4
                    }

                    # Xác định số kênh đầu vào dựa trên dataset
                    if "MNIST" in args.dataset or "FashionMNIST" in args.dataset:
                        input_channels = 1
                    elif "Cifar10" in args.dataset:
                        input_channels = 3
                    else:
                        # Các dataset khác có thể cần xử lý riêng hoặc báo lỗi
                        raise NotImplementedError(
                            f"Hybrid_QCNN not specifically configured for dataset {args.dataset}. Check input channels.")

                    # Khởi tạo model, truyền vào num_classes, weight_shapes và in_channels
                    args.model = Hybrid_QCNN(
                        num_classes=args.num_classes,
                        weight_shapes=hybrid_weight_shapes,
                        in_channels=input_channels  # <<< Truyền số kênh vào đây
                    ).to(args.device)
        elif model_str == "Quanv_torch":
            if "MNIST" in args.dataset or "FashionMNIST" in args.dataset:
                    input_channels = 1
            elif "Cifar10" in args.dataset or "Cifar100" in args.dataset:
                    input_channels = 3
            else:
                    raise NotImplementedError(
                        f"Quanv not specifically configured for dataset {args.dataset}."
                    )

            args.model = Hybrid_QCNN_TQ(
                    num_classes=args.num_classes,
                    in_channels=input_channels
                ).to(args.device)
        

        else:
            raise Exception("Model not found")

        print(args.model)
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)
        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()

            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)
        elif args.algorithm == "FedAvg-WA": # Thuật toán Weighted Averaging
            server = FedAvgWA(args, i)
        elif args.algorithm == "FedAvg-MD": # Thuật toán Model Drift
            server = FedAvgMD(args, i)
        else:
            raise Exception("Algorithm not found")

        # Debug: Check data consistency
        # for client in server.clients:
        #     print(f"Client {client.id} - Number of samples: {len(client.train_data)}")

        server.train()
        time_list.append(time.time() - start)
    print(f"\nAvergae time cost:{round(np.average(time_list), 2)}s.")

    average_data(dataset=args.dataset, algorithm=args.algorithm, model_name=args.model_name, times=args.times)
    print("done")
    reporter.report()


if __name__ == "__main__":
    total_start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-alpha', "--alpha", type=float, default=0.1,
                    help="Weight for global model in weighted_avg")
    parser.add_argument('-omega', "--omega", type=float, default=0.9,
                    help="Weight for local model in weighted_avg")
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="MNIST")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="CNN")
    parser.add_argument('-m_name', "--model_name", type=str, default="CNN")
    parser.add_argument('-lbs', "--batch_size", type=int, default=10)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2000)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100,
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=32000,
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    parser.add_argument('-kUe', "--num_terminals", type=int, default=40, help="Number of terminals (K)")
    parser.add_argument('-mm', "--massive_mimo", type=bool, default=False,
                        help="Run in massive MIMO scenario")
    parser.add_argument('-pn', "--pilot_num", type=int, default=20, help="Number of pilots (tau)")  # Add this argument
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=', getattr(args, arg))
    print("=" * 50)
    run(args)

