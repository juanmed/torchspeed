import numpy as np
import cupy as cp

x_gpu = cp.array([1,2,3])  # an cupy array created in GPU
x_cpu = np.array([1,2,3])  # an numpy array created in CPU

# See current default device, and see a variable's current device
print("Current device: {}".format(cp.cuda.Device()))
print("x_gpu norm: {}, {}".format(cp.linalg.norm(x_gpu), x_gpu.device))
print("x_cpu norm: {}".format(np.linalg.norm(x_cpu)))

# Change CUDA Device temporarily
# throws error if device unavailable
try:
    with cp.cuda.Device(1):
        y_gpu = cp.array([2,3,4])
except:
    print("Device unavailable")

# Change CUDA Device permanently
cp.cuda.Device(0).use()
z_gpu = cp.array([4,5,6])

# Move array between devices
# This means we can move from Device 0 to Device 1
try:
    with cp.cuda.Device(1):
        k_gpu = cp.asarray(z_gpu)
except:
    print("Device Unavailable")

# MOve array from device to host
a_cpu = cp.asnumpy(z_gpu)
b_cpu = z_gpu.get()

print("a_cpu = cp.asnumpy(z_gpu) : {},{}".format(a_cpu, type(a_cpu)))
print("b_cpu = z_gpu.get() : {},{}".format(b_cpu, type(b_cpu)))






