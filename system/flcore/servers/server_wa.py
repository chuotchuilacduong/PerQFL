# flcore/servers/server_wa.py

import time
from flcore.servers.serverbase import Server
from flcore.clients.client_wa import clientWA

class FedAvgWA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # Các hàm này được lấy từ __init__ của FedAvg
        self.set_slow_clients()
        self.set_clients(clientWA)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients for Weighted-Averaging.")

        self.Budget = []

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round {i+1}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            # --- PHẦN LOGIC ĐÃ ĐIỀU CHỈNH CHO WA ---
            for client in self.selected_clients:
                # 1. Client thực hiện cá nhân hóa model
                client.personalize_model(self.global_model)
                # 2. Client huấn luyện trên model đã cá nhân hóa
                client.train()
            # -----------------------------------------

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()