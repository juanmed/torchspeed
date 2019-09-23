import cupy as cp
import SpeedTorch
import torch
import numpy as np 
import torch.nn as nn
import os, signal


torch.set_default_dtype(torch.float32)
#torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# create a cuda tensor
xgpu = torch.zeros(6,3).to(device)


sampl = np.random.uniform(low=-1.0, high=1.0, size=(6, 3))
np.save('data.npy', sampl)
del sampl

empty = np.zeros((6,3))
data = np.random.rand(6,3)
#with data as file:
cpu_pinned1 = SpeedTorch.DataGadget(data, CPUPinn = True )
cpu_pinned1.gadgetInit()
print("Device: ", cpu_pinned1.CUPYcorpus.device)


cpu_pinned2 = SpeedTorch.DataGadget(empty, CPUPinn = True )
cpu_pinned2.gadgetInit()

# transfer CPU Pinned SpeedTorch Object to Pytorch GPU tensor
print("X GPU before transfer: \n{}".format(xgpu))
xgpu[:] =cpu_pinned1.getData( indexes = (slice(0,6),slice(0,3)) )
print("X GPU after transfer: \n{}".format(xgpu))


# transfer Pytorch GPU Tensor back to CPU Pinned SpeedTorch object
print("CPU Pinned before reception: \n{}".format(cpu_pinned2.CUPYcorpus))
cpu_pinned2.insertData(dataObject = xgpu, indexes=(slice(0,6), slice(0,3)))
print("CPU Pinned after reception: \n{}".format(cpu_pinned2.CUPYcorpus))

# Kill process on exit
print("\nKilling Process... bye!")
os.kill(os.getpid(), signal.SIGKILL)