import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import cupy as cp
import os, signal
import SpeedTorch



torch.set_default_dtype(torch.float64)
#torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

fig = plt.figure(figsize = (20,10))
ax = fig.add_subplot(1,1,1)
cpu = 'Intel Core i5-8400 CPU @ 2.80GHz x 6'
gpu = 'GeForce GTX 1050 Ti/PCIe/SSE2'

a = np.array([1.,1.,1.], dtype = np.float16)
b = np.array([1.,2.,3.], dtype = np.float32)
c = np.array([1.,2.,3.], dtype = np.float64)

Ra = np.random.rand(3,3).astype(np.float16)
Rb = np.random.rand(3,3).astype(np.float32)
Rc = np.random.rand(3,3).astype(np.float64)
print(Ra.dtype)

x = np.float16(np.random.rand())
#y = np.random.rand().astype(np.float32)
#z = np.random.rand().astype(np.float64)
#print(x,y,z)

loops = 10000
array_trans_times = []
value_trans_times = []
matrix_trans_times = []
direct_trans_times = []
gpu_matrix_trans_times = []
vargpu_matrix_trans_times = []
constgpu_matrix_trans_times = []
cupy_matrix_create_times = []
cpu_pinned_create_times = []
storch_cpu_pinned_create_times = []

#warm up
q = np.random.rand(1000,1000)
t = np.random.rand(1000,1000)
np.dot(q,t)
q = torch.from_numpy(q).to(device)
t = torch.from_numpy(t).to(device)
torch.mm(q,t)
cp.matmul(cp.asarray(q),cp.asarray(t))

m = 3.2153548435
n = 1.5484574874
r = 4.1545348574

t = np.float64

b = np.random.rand(6,3).astype(t)
pytorchCPUpinned = torch.from_numpy(b).pin_memory() 


#
#   Speed Torch test variables
#

# create a cuda tensor
xgpu = torch.zeros(6,3).to(device)
data = np.random.rand(6,3)
cpu_pinned1 = SpeedTorch.DataGadget(data, CPUPinn = True )
cpu_pinned1.gadgetInit()
xgpu[:] =cpu_pinned1.getData( indexes = (slice(0,6),slice(0,3)) )  # transfer cpu_pinned to xgpu

for i in range(loops):

    #data = np.random.rand(6,3)
    with torch.no_grad():


        # transfer 6 3xnp_arrays to device
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        a = np.random.rand(3).astype(t)
        torch.cuda.synchronize()
        start.record()
        a_cuda = torch.from_numpy(a).to(device)
        a_cuda = torch.from_numpy(a).to(device)
        a_cuda = torch.from_numpy(a).to(device)
        a_cuda = torch.from_numpy(a).to(device)
        a_cuda = torch.from_numpy(a).to(device)
        a_cuda = torch.from_numpy(a).to(device)
        end.record()
        torch.cuda.synchronize()
        array_trans_times.append(start.elapsed_time(end))

        # transfer one 6x3 np.array to device
        start2 = torch.cuda.Event(enable_timing=True)
        end2 = torch.cuda.Event(enable_timing=True)
        b = np.random.rand(6,3).astype(t)
        torch.cuda.synchronize()
        start2.record()
        b_cuda = torch.from_numpy(b).to(device)
        end2.record()
        torch.cuda.synchronize()
        matrix_trans_times.append(start2.elapsed_time(end2))

        #Create 6 3xarrays in device
        start3 = torch.cuda.Event(enable_timing=True)
        end3 = torch.cuda.Event(enable_timing=True)
        x = np.float64(np.random.rand())#.astype(np.float16)
        y = np.float64(np.random.rand())#.astype(np.float32)
        z = np.float64(np.random.rand())#.astype(np.float64)
        torch.cuda.synchronize()
        start3.record()
        c_cuda = torch.tensor([x,y,z], device =device)
        c_cuda = torch.tensor([x,y,z], device =device)
        c_cuda = torch.tensor([x,y,z], device =device)
        c_cuda = torch.tensor([x,y,z], device =device)
        c_cuda = torch.tensor([x,y,z], device =device)
        c_cuda = torch.tensor([x,y,z], device =device)
        end3.record()
        torch.cuda.synchronize()
        value_trans_times.append(start3.elapsed_time(end3))
        
        # create 6 3xarrays in device
        start4 = torch.cuda.Event(enable_timing=True)
        end4 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start4.record()
        d_cuda = torch.tensor([3.2153548435,1.5484574874,4.1545348574], device = device)
        d_cuda = torch.tensor([3.2153548435,1.5484574874,4.1545348574], device = device)
        d_cuda = torch.tensor([3.2153548435,1.5484574874,4.1545348574], device = device)
        d_cuda = torch.tensor([3.2153548435,1.5484574874,4.1545348574], device = device)
        d_cuda = torch.tensor([3.2153548435,1.5484574874,4.1545348574], device = device)
        d_cuda = torch.tensor([3.2153548435,1.5484574874,4.1545348574], device = device)
        end4.record()
        torch.cuda.synchronize()
        direct_trans_times.append(start4.elapsed_time(end4))

        # Create one 6x3 tensor in device
        start5 = torch.cuda.Event(enable_timing=True)
        end5 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start5.record()
        f_cuda = torch.rand(6,3, device = device)
        end5.record()
        torch.cuda.synchronize()
        gpu_matrix_trans_times.append(start5.elapsed_time(end5)) 

        start6 = torch.cuda.Event(enable_timing=True)
        end6 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start6.record()
        g_cuda = torch.tensor([[x,y,z],
                            [x,y,z],
                            [x,y,z],
                            [x,y,z],
                            [x,y,z],
                            [x,y,z]], device = device)
        end6.record()
        torch.cuda.synchronize()
        vargpu_matrix_trans_times.append(start6.elapsed_time(end6)) 

        # create a 6x3 tensor in device
        start7 = torch.cuda.Event(enable_timing=True)
        end7 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start7.record()
        h_cuda = torch.tensor([[3.2153548435,1.5484574874,4.1545348574],
                            [3.2153548435,1.5484574874,4.1545348574],
                            [3.2153548435,1.5484574874,4.1545348574],
                            [3.2153548435,1.5484574874,4.1545348574],
                            [3.2153548435,1.5484574874,4.1545348574],
                            [3.2153548435,1.5484574874,4.1545348574]], device = device)
        end7.record()
        torch.cuda.synchronize()
        constgpu_matrix_trans_times.append(start7.elapsed_time(end7))

        # create a 6x3 cupy array in device
        start8 = torch.cuda.Event(enable_timing=True)
        end8 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        cp.cuda.Device().synchronize()
        start8.record()
        i_cuda = cp.random.rand(6,3)
        end8.record()
        cp.cuda.Device().synchronize()
        torch.cuda.synchronize()
        cupy_matrix_create_times.append(start8.elapsed_time(end8))

        # create 6x3 pinned CPU tensor and transfer to GPU
        start9 = torch.cuda.Event(enable_timing=True)
        end9 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start9.record()
        j_cuda = pytorchCPUpinned.to(device)
        end9.record()
        torch.cuda.synchronize()
        cpu_pinned_create_times.append(start9.elapsed_time(end9))    

        # create 6x3 pinned CPU tensor and transfer to GPU
        start9 = torch.cuda.Event(enable_timing=True)
        end9 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start9.record()
        j_cuda = pytorchCPUpinned.to(device)
        #excess = j_cuda.cpu()
        end9.record()
        torch.cuda.synchronize()
        cpu_pinned_create_times.append(start9.elapsed_time(end9))

        # transfer a 6x3 speedtorch slice to a Pytorch GPU Tensor
        start10 = torch.cuda.Event(enable_timing=True)
        end10 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start10.record()
        xgpu[:] =cpu_pinned1.getData( indexes = (slice(0,6),slice(0,3)) )
        #cpu_pinned1.insertData(dataObject = xgpu, indexes=(slice(0,6), slice(0,3)))
        end10.record()
        torch.cuda.synchronize()
        storch_cpu_pinned_create_times.append(start10.elapsed_time(end10))


logx = True
logy = False
array_times = pd.DataFrame(array_trans_times, columns = ['6x 3-np.array'])
array_times['6x 3-np.array'] = array_times['6x 3-np.array']
ax = array_times.plot.hist(ax = ax, bins = 400, logx = logx, logy=logy, title = 'Transfer Times for float64\n{} CPU\n {} GPU'.format(cpu,gpu), color = 'r', alpha  = 0.5)#, xlim = (0.01,1.0))
ax.set_xlabel("CPU -> GPU Transfer times {ms}")
print("Numpy Times: points: {}, mean: {}, std: {}, max: {}, min: {}".format(len(array_trans_times), np.mean(array_trans_times), np.std(array_trans_times), np.max(array_trans_times), np.min(array_trans_times)))

matrix_times = pd.DataFrame(matrix_trans_times, columns = ['6x3np.array'])
matrix_times['6x3np.array'] = matrix_times['6x3np.array']
matrix_times.plot.hist(ax = ax, bins = 400, logx = logx, logy=logy, color = 'g', alpha  = 0.5)
print("Matrix Times: points: {}, mean: {}, std: {}, max: {}, min: {}".format(len(matrix_trans_times), np.mean(matrix_trans_times), np.std(matrix_trans_times), np.max(matrix_trans_times), np.min(matrix_trans_times)))

value_times = pd.DataFrame(value_trans_times, columns = ['6x 3-tensor'])
value_times['6x 3-tensor'] = value_times['6x 3-tensor']
value_times.plot.hist(ax = ax, bins = 400, logx = logx, logy=logy, color = 'b', alpha  = 0.5)
print("Scalar Times: points: {}, mean: {}, std: {}, max: {}, min: {}".format(len(value_trans_times), np.mean(value_trans_times), np.std(value_trans_times), np.max(value_trans_times), np.min(value_trans_times)))

direct_times = pd.DataFrame(direct_trans_times, columns = ['6x 3-tensor(const)'])
direct_times['6x 3-tensor(const)'] = direct_times['6x 3-tensor(const)']
direct_times.plot.hist(ax = ax, bins = 400, logx = logx, logy=logy, color = 'm', alpha  = 0.5)
print("Direct Times: points: {}, mean: {}, std: {}, max: {}, min: {}".format(len(direct_trans_times), np.mean(direct_trans_times), np.std(direct_trans_times), np.max(direct_trans_times), np.min(direct_trans_times)))

gpumatrix_times = pd.DataFrame(gpu_matrix_trans_times, columns = ['6x3 tensor(rand)'])
gpumatrix_times['6x3 tensor(rand)'] = gpumatrix_times['6x3 tensor(rand)']
gpumatrix_times.plot.hist(ax = ax, bins = 400, logx = logx, logy=logy, color = 'c', alpha  = 0.5)
print("GPUMat Times: points: {}, mean: {}, std: {}, max: {}, min: {}".format(len(gpu_matrix_trans_times), np.mean(gpu_matrix_trans_times), np.std(gpu_matrix_trans_times), np.max(gpu_matrix_trans_times), np.min(gpu_matrix_trans_times)))


vargpumatrix_times = pd.DataFrame(vargpu_matrix_trans_times, columns = ['6x3 tensor(var)'])
vargpumatrix_times['6x3 tensor(var)'] = vargpumatrix_times['6x3 tensor(var)']
vargpumatrix_times.plot.hist(ax = ax, bins = 400, logx = logx, logy=logy, color = 'yellow', alpha  = 0.5)
print("VGPUMat Times: points: {}, mean: {}, std: {}, max: {}, min: {}".format(len(vargpu_matrix_trans_times), np.mean(vargpu_matrix_trans_times), np.std(vargpu_matrix_trans_times), np.max(vargpu_matrix_trans_times), np.min(gpu_matrix_trans_times)))

constgpumatrix_times = pd.DataFrame(constgpu_matrix_trans_times, columns = ['6x3 tensor(const)'])
constgpumatrix_times['6x3 tensor(const)'] = constgpumatrix_times['6x3 tensor(const)']
constgpumatrix_times.plot.hist(ax = ax, bins = 400, logx = logx, logy=logy, color = 'chartreuse', alpha  = 0.5)
print("CGPUMat Times: points: {}, mean: {}, std: {}, max: {}, min: {}".format(len(constgpu_matrix_trans_times), np.mean(constgpu_matrix_trans_times), np.std(constgpu_matrix_trans_times), np.max(constgpu_matrix_trans_times), np.min(constgpu_matrix_trans_times)))

cupymatrix_times = pd.DataFrame(cupy_matrix_create_times, columns = ['6x3 cupy(rand)'])
cupymatrix_times['6x3 cupy(rand)'] = cupymatrix_times[cupymatrix_times['6x3 cupy(rand)'] < 1.0]
cupymatrix_times.plot.hist(ax = ax, bins = 400, logx = logx, logy=logy, color = 'orange', alpha  = 0.5)
print("CUPYMat Times: points: {}, mean: {}, std: {}, max: {}, min: {}".format(len(cupy_matrix_create_times), np.mean(cupy_matrix_create_times), np.std(cupy_matrix_create_times), np.max(cupy_matrix_create_times), np.min(cupy_matrix_create_times)))

cpu_pinned_matrix_times = pd.DataFrame(cpu_pinned_create_times, columns = ['6x3 cpu pinned'])
cpu_pinned_matrix_times['6x3 cpu pinned'] = cpu_pinned_matrix_times[cpu_pinned_matrix_times['6x3 cpu pinned'] < 1.0]
cpu_pinned_matrix_times.plot.hist(ax = ax, bins = 400, logx = logx, logy=logy, color = 'pink', alpha  = 0.5)
print("CPU Pinned Times: points: {}, mean: {}, std: {}, max: {}, min: {}".format(len(cpu_pinned_create_times), np.mean(cpu_pinned_create_times), np.std(cpu_pinned_create_times), np.max(cpu_pinned_create_times), np.min(cpu_pinned_create_times)))


cpu_pinned_st_matrix_times = pd.DataFrame(storch_cpu_pinned_create_times, columns = ['6x3 CPUPinn ST'])
cpu_pinned_st_matrix_times['6x3 CPUPinn ST'] = cpu_pinned_st_matrix_times[cpu_pinned_st_matrix_times['6x3 CPUPinn ST'] < 1.0]
cpu_pinned_st_matrix_times.plot.hist(ax = ax, bins = 400, logx = logx, logy=logy, color = 'turquoise', alpha  = 0.5)
print("ST CPU Pinned: points: {}, mean: {}, std: {}, max: {}, min: {}".format(len(storch_cpu_pinned_create_times), np.mean(storch_cpu_pinned_create_times), np.std(storch_cpu_pinned_create_times), np.max(storch_cpu_pinned_create_times), np.min(storch_cpu_pinned_create_times)))


plt.show()

# Kill process on exit
print("\nKilling Process... bye!")
os.kill(os.getpid(), signal.SIGKILL)





"""
a = torch.randn(...)
torch.cuda.synchronize()
t0 = time.time()
a = a.to('cuda:0')
torch.cuda.synchronize()
t1 = time.time()

    ns=[600,6000,6000,60000,60000]
    for i, p in enumerate(ps):
        n=ns[i]
        A=torch.rand(p,n).cpu()
        torch.cuda.synchronize()
        t0 = time.time()
        A.qr()
        torch.cuda.synchronize()
        t1 = time.time()
        t=t1-t0
        print('%d*%d cpu: time_cost %f'%(p,n,t))
        A=torch.rand(p,n).cuda()
        torch.cuda.synchronize()
        t0 = time.time()
        A.qr()
        torch.cuda.synchronize()
        t1 = time.time()
        t=t1-t0
        print('%d*%d cuda: time_cost %f'%(p,n,t))
        A = torch.randn(p,n).cpu()
        torch.cuda.synchronize()
        t0 = time.time()
        A = A.to('cuda:0')
        torch.cuda.synchronize()

"""