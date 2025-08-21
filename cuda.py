import torch

print("CUDA Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0))
print("GPU Memory Allocated:", torch.cuda.memory_allocated(0), "bytes")
