import torch

# Load model state_dict
model_path = r"G:\BK\Quantum\cfmimo_fed_qcnn\quantum_classical_comparison\best_Quantum_Hybrid_(PCA).pth"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))  # Load OrderedDict

# Kiểm tra danh sách các layer
print(state_dict.keys())

# In trọng số từng layer
for name, param in state_dict.items():
    print(f"Layer: {name} | Shape: {param.shape}")
    print(param)  # In toàn bộ trọng số
