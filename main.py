import torch
print(torch.cuda.is_available())      # True si CUDA est détecté
print(torch.cuda.get_device_name(0))  # Affiche le nom de ton GPU