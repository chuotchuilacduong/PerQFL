import torch
import torch.nn.functional as F
from torch import nn
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import torchquantum as tq
import torchquantum.functional as tqf
batch_size = 32

n_qubits = 8

class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out


class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(inplace=True)
        )
        self.fc2=  nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        self.fc3= nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)  # [batch_size, 64, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        x= self.fc3(x)
        x = self.fc(x)
        return x

class FedAvgMLP(nn.Module):
    def __init__(self, in_features=3072, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 350)
        self.fc2 = nn.Linear(350, 50)
        self.fc3 = nn.Linear(50, 350)
        self.fc = nn.Linear(350, num_classes) 
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))    
        x = self.act(self.fc3(x))
        x = self.fc(x)
        return x

def U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6,
          wires):
    qml.U3(*weights_0, wires=wires[0])
    qml.U3(*weights_1, wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(weights_2, wires=wires[0])
    qml.RZ(weights_3, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(weights_4, wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(*weights_5, wires=wires[0])
    qml.U3(*weights_6, wires=wires[1])


# @qml.qnode(dev1)
def qnode(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6):
    # Amplitude Embedding
    qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True, pad_with=0)
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[0, 1])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[2, 3])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[4, 5])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[6, 7])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[1, 2])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[3, 4])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[5, 6])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[7, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class HQCNN_Ang_noQP(nn.Module):
    def __init__(self, in_features, num_classes, weight_shapes):
        super(HQCNN_Ang_noQP, self).__init__()
        self.classical_layer_1 = nn.Linear(in_features, 512)
        self.fc1 = nn.Linear(512, n_qubits)
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.fc = nn.Linear(n_qubits, num_classes)

    def forward(self, x):
        # Flatten input if not already
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = self.classical_layer_1(x)  # Nhân input với weight của classical_layer_1
        x = self.fc1(x)
        x = self.quantum_layer(x)  # Qua quantum layer
        x = self.fc(x)
        return x



n_qubits_hybrid = 4  
dev_hybrid = qml.device("default.qubit", wires=n_qubits_hybrid)
dev = qml.device("default.qubit", wires=n_qubits_hybrid, shots = 1000)


@qml.qnode(dev_hybrid, interface='torch')
def hybrid_quantum_circuit(inputs,
                           weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6,
                           weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13): # Added new weights

    # qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits_hybrid), normalize=True, pad_with=0.)
    qml.AngleEmbedding(inputs,wires=range(n_qubits_hybrid),rotation='Y')
    # pairs = [(0, 1), (1, 2), (2, 3), (3, 4),(4,5),(5,0)]  # Ring coupling for 6 qubits
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Ring coupling for 4 qubits

    # First U_SU4 Layer
    for i, wires_pair in enumerate(pairs):
        U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=wires_pair)
    # Second U_SU4 Layer with new weights
    for i, wires_pair in enumerate(pairs):
        U_SU4(weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13, wires=wires_pair) # Using new weights

    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits_hybrid)]


class HybridQuanvLayer(nn.Module):
    def __init__(self, weight_shapes):
        super().__init__()
        self.n_qubits = n_qubits_hybrid  
        self.qlayer = qml.qnn.TorchLayer(hybrid_quantum_circuit, weight_shapes)

    def forward(self, patches):
        # patches shape: [batch * n_patches, patch_size]
        batch_x_n_patches, patch_size = patches.shape
        inputs_scaled= torch.sigmoid(patches)*np.pi
        quantum_output = self.qlayer(inputs_scaled)

        #######
        # amplitude
        # epsilon = 0
        # processed_patches = patches.float() + epsilon
        # quantum_output = self.qlayer(processed_patches)
        return quantum_output



def hybrid_extract_patches(feature_map, kernel_size=2, stride=2):
    # feature_map shape: (batch, channels, height, width)
    patches = feature_map.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    # patches shape: (batch, channels, n_h, n_w, k1, k2)
    b, c, n_h, n_w, k1, k2 = patches.shape

    expected_patch_size = c * k1 * k2
    return patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(b, n_h * n_w, expected_patch_size)


class Hybrid_QCNN(nn.Module):
    def __init__(self, num_classes=10, weight_shapes=None,
                 in_channels=1):  
        super().__init__()
        self.n_qubits = n_qubits_hybrid  
        self.input_channels = in_channels
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [B, 4, 14, 14]
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [B, 8, 7, 7]

            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1),  # Output: [B, 4, 7, 7]
            nn.LeakyReLU()
        )
        _dummy_input_size = 32 if self.input_channels == 3 else 28  
        with torch.no_grad():
            _dummy_input = torch.zeros(1, self.input_channels, _dummy_input_size, _dummy_input_size)
            _cnn_out_shape = self.cnn_layers(_dummy_input).shape
            self.cnn_output_height = _cnn_out_shape[2] 
            self.cnn_output_width = _cnn_out_shape[3]

        self.q_layer = HybridQuanvLayer(weight_shapes=weight_shapes)
        self.patch_kernel = 1 
        self.patch_stride = 1
        n_patches_h = (self.cnn_output_height - self.patch_kernel) // self.patch_stride + 1  # 7
        n_patches_w = (self.cnn_output_width - self.patch_kernel) // self.patch_stride + 1  # 7
        num_patches = n_patches_h * n_patches_w  
        fc_input_size = num_patches * self.n_qubits  

        self.fc1 = nn.Sequential(
            nn.Linear(fc_input_size, 64), 
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64), 
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        cnn_out = self.cnn_layers(x)  # [B, 4, 7, 7]
        patches = hybrid_extract_patches(cnn_out, kernel_size=self.patch_kernel, stride=self.patch_stride)  # [B, 49, 4]
        b, n_p, p_f = patches.shape
        patches_flat = patches.view(-1, p_f)  # [B * 49, 4]
        q_out_flat = self.q_layer(patches_flat)  # [B * 49, 4]
        q_out = q_out_flat.view(b, n_p, self.n_qubits)  # [B, 49, 4]
        q_out_flattened = q_out.view(q_out.size(0), -1)  # [B, 196]
        out=self.fc1(q_out_flattened)
        output = self.fc(out)  # [B, num_classes]
        return output
class TwoQubitAnsatz(tq.QuantumModule):
   
    def __init__(self):
        super().__init__()
        # Định nghĩa các cổng sẽ được sử dụng
        self.ry = tq.RY()
        self.rz = tq.RZ()
        self.cnot = tq.CNOT()

    def forward(self, q_device: tq.QuantumDevice, params: torch.Tensor, wires: list[int]):
      
        self.ry(q_device, wires=wires[0], params=params[0])
        self.rz(q_device, wires=wires[0], params=params[1])
        self.ry(q_device, wires=wires[1], params=params[2])
        self.rz(q_device, wires=wires[1], params=params[3])

        self.cnot(q_device, wires=[wires[0], wires[1]])

        self.ry(q_device, wires=wires[0], params=params[4])
        self.rz(q_device, wires=wires[0], params=params[5])
        self.ry(q_device, wires=wires[1], params=params[6])


class TorchQuantumCircuit(tq.QuantumModule):
   
    def __init__(self, n_qubits=4):
        super().__init__()
        self.n_wires = n_qubits
        self.pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Ring coupling

        self.params = nn.Parameter(torch.rand(2, 7))

        self.encoder = tq.GeneralEncoder(
            [{"input_idx": i, "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )
        self.ansatz_layer = TwoQubitAnsatz()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor):
       
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=False
        )

        self.encoder(qdev, x)

        for pair in self.pairs:
            self.ansatz_layer(qdev, self.params[0], wires=pair)

        for pair in self.pairs:
            self.ansatz_layer(qdev, self.params[1], wires=pair)
        return self.measure(qdev)


class HybridQuanvLayerTQ(nn.Module):
    def __init__(self, n_qubits=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layer = TorchQuantumCircuit(n_qubits=self.n_qubits)

    def forward(self, patches):
       
        inputs_scaled = torch.sigmoid(patches) * np.pi
        
        quantum_output = self.q_layer(inputs_scaled)
        return quantum_output


def hybrid_extract_patches(feature_map, kernel_size=1, stride=1):
    patches = feature_map.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    b, c, n_h, n_w, k1, k2 = patches.shape
    expected_patch_size = c * k1 * k2
    return patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(b, n_h * n_w, expected_patch_size)


class Hybrid_QCNN_TQ(nn.Module):
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.n_qubits = 4 
        self.input_channels = in_channels
        
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=self.n_qubits, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        
       
        _dummy_input_size = 32 if self.input_channels == 3 else 28 
        with torch.no_grad():
            _dummy_input = torch.zeros(1, self.input_channels, _dummy_input_size, _dummy_input_size)
            _cnn_out_shape = self.cnn_layers(_dummy_input).shape
            self.cnn_output_height = _cnn_out_shape[2]
            self.cnn_output_width = _cnn_out_shape[3]

        self.q_layer = HybridQuanvLayerTQ(n_qubits=self.n_qubits)

        self.patch_kernel = 1
        self.patch_stride = 1
        n_patches_h = (self.cnn_output_height - self.patch_kernel) // self.patch_stride + 1
        n_patches_w = (self.cnn_output_width - self.patch_kernel) // self.patch_stride + 1
        num_patches = n_patches_h * n_patches_w
        
        fc_input_size = num_patches * self.n_qubits
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )


    def forward(self, x):
        cnn_out = self.cnn_layers(x)  # [B, 4, 7, 7]

        patches = hybrid_extract_patches(cnn_out, kernel_size=self.patch_kernel, stride=self.patch_stride)  # [B, 49, 4]
        b, n_p, p_f = patches.shape
        patches_flat = patches.view(-1, p_f)  
        
        q_out_flat = self.q_layer(patches_flat)  
        q_out = q_out_flat.view(b, n_p, self.n_qubits)  # [B, 49, 4]

        q_out_flattened = q_out.view(q_out.size(0), -1)  # [B, 196]
        output = self.fc(q_out_flattened)
        
        return output