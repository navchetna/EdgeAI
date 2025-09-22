import torch
tensor_1 = torch.randn(1, 1, 40, 128).to('xpu')
tensor_2 = torch.randn(1, 1, 128, 40).to('xpu')
print(torch.matmul(tensor_1, tensor_2).size())