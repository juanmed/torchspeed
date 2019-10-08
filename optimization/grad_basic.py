import torch
from torchviz import make_dot

use_gpu = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
N, D = 1000,3

q = torch.rand(N, D).type(dtype)
p = torch.rand(N, D).type(dtype)
s = torch.tensor([2.5]).type(dtype)

q.requires_grad = True
p.requires_grad = True

q_i = q[:,None,:]
q_j = q[None,:,:]
D_ij = ((q_i - q_j)**2).sum(dim=2)
K_ij = (-D_ij/(2*s**2)).exp()
v = K_ij @ p

H = .5 * torch.dot(p.view(-1), v.view(-1))

make_dot(H, {'q':q, 'p':p}).render(view=True)