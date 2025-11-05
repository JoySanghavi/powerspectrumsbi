import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Number of GPUs PyTorch sees
print(torch.cuda.get_device_name(0))  # Name of first GPU
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
print(torch.__version__)
