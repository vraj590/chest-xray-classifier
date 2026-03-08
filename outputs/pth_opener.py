import torch

file_path = "best_model.pth"

loaded_data = torch.load(file_path, weights_only=False)

print(type(loaded_data))

if isinstance(loaded_data, dict):
    print(loaded_data.keys())