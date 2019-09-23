import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

@torch.jit.script
def addcmul(i, v, t1, s, r):
    # type: (Tensor, float, Tensor, Tensor, Tensor) -> Tensor

    return i.addcmul(tensor1 = t1, tensor2 = s - r, value =v).sum(dim = 0)

def addcmulp(i, v, t1, s, r):
    # type: (Tensor, float, Tensor, Tensor) -> Tensor
    
    return torch.addcmul(i, v, t1, s - r).sum(dim = 0)


def main():
    #torch.set_default_tensor_type(torch.FloatTensor)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    cpu = 'Intel Core i5-8400 CPU @ 2.80GHz x 6'
    gpu = 'GeForce GTX 1050 Ti/PCIe/SSE2'

    a = torch.tensor([1.,1.,1.], dtype = torch.float64).to(device)
    b = torch.tensor([1.,2.,3.], dtype = torch.float64).to(device)
    c = torch.tensor([1.,2.,3.], dtype = torch.float64).to(device)

    Ra = np.random.rand(3,3).astype(np.float16)
    Rb = np.random.rand(3,3).astype(np.float32)
    Rc = np.random.rand(3,3).astype(np.float64)

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
    
    #warm up
    q = np.random.rand(1000,1000)
    t = np.random.rand(1000,1000)
    np.dot(q,t)
    q = torch.from_numpy(q).to(device)
    t = torch.from_numpy(t).to(device)
    torch.mm(q,t)

    m = 3.2153548435
    n = 1.5484574874
    r = 4.1545348574

    t = np.float64
    kp = 7.0
    ki = 2.0
    kd = 4.0

    a = torch.zeros(5,3).to(device)
    k = torch.tensor([[kp, kp, kp],
                      [ki, ki, ki],
                      [kd, kd, kd],
                      [0.0,0.0,0.0],
                      [0.0,0.0,0.0]]).to(device)
    s = torch.ones(5,3).to(device)*2.
    r = torch.ones(5,3).to(device)*1.


    Kp_gains = k[0]
    Kd_gains = k[1]
    Ki_gains = k[2]
    Kj_gains = k[3]
    Kz_gains = k[4]

    p = s[0]
    v = s[1]
    ac = s[2]
    j = s[3]
    sn = s[4]

    pr = r[0]
    vr = r[1]
    ar = r[2]
    jr = r[3]
    sr = r[4]

    for i in range(loops):
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        #print("addcmul jit", addcmul(a,-1.0,k,s,r))
        addcmul(a,-1.0,k,s,r)
        end.record()
        torch.cuda.synchronize()
        array_trans_times.append(start.elapsed_time(end))

        
        start2 = torch.cuda.Event(enable_timing=True)
        end2 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start2.record()
        #print("addcmul py", addcmulp(a,-1.0,k,s,r))
        addcmulp(a,-1.0,k,s,r)
        end2.record()
        torch.cuda.synchronize()
        matrix_trans_times.append(start2.elapsed_time(end2))

        Kp_gains = k[0]
        Kd_gains = k[1]
        Ki_gains = k[2]
        Kj_gains = k[3]
        Kz_gains = k[4]

        p = s[0]
        v = s[1]
        ac = s[2]
        j = s[3]
        sn = s[4]

        pr = r[0]
        vr = r[1]
        ar = r[2]
        jr = r[3]
        sr = r[4]


        start3 = torch.cuda.Event(enable_timing=True)
        end3 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start3.record()
        a_e = torch.zeros(3).to(device)
        a_e = torch.addcmul(a_e, -1.0, Kp_gains, torch.add(p,-1.0,pr))#.type(torch.cuda.FloatTensor))
        a_e = torch.addcmul(a_e, -1.0, Kd_gains, torch.add(v,-1.0,vr))#.type(torch.cuda.FloatTensor))
        a_e = torch.addcmul(a_e, -1.0, Ki_gains, torch.add(ac,-1.0,ar))
        a_e = torch.addcmul(a_e, -1.0, Kj_gains, torch.add(j,-1.0,jr))
        a_e = torch.addcmul(a_e, -1.0, Kz_gains, torch.add(sn,-1.0,sr))
        #print("addcmulx5 gpu",a_e)
        end3.record()
        torch.cuda.synchronize()
        value_trans_times.append(start3.elapsed_time(end3))


        Kp_gains = k[0].cpu()
        Kd_gains = k[1].cpu()
        Ki_gains = k[2].cpu()
        Kj_gains = k[3].cpu()
        Kz_gains = k[4].cpu()

        p = s[0].cpu()
        v = s[1].cpu()
        ac = s[2].cpu()
        j = s[3].cpu()
        sn = s[4].cpu()

        pr = r[0].cpu()
        vr = r[1].cpu()
        ar = r[2].cpu()
        jr = r[3].cpu()
        sr = r[4].cpu()  
        
        start4 = torch.cuda.Event(enable_timing=True)
        end4 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start4.record()
        a_e = torch.zeros(3)
        a_e = torch.addcmul(a_e, -1.0, Kp_gains, torch.add(p,-1.0,pr))#.type(torch.cuda.FloatTensor))
        a_e = torch.addcmul(a_e, -1.0, Kd_gains, torch.add(v,-1.0,vr))#.type(torch.cuda.FloatTensor))
        a_e = torch.addcmul(a_e, -1.0, Ki_gains, torch.add(ac,-1.0,ar))
        a_e = torch.addcmul(a_e, -1.0, Kj_gains, torch.add(j,-1.0,jr))
        a_e = torch.addcmul(a_e, -1.0, Kz_gains, torch.add(sn,-1.0,sr))
        #print("addcmulx5 cpu", a_e)
        end4.record()
        torch.cuda.synchronize()
        direct_trans_times.append(start4.elapsed_time(end4))

        """
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
        f_cuda = torch.tensor([[x,y,z],
                            [x,y,z],
                            [x,y,z],
                            [x,y,z],
                            [x,y,z],
                            [x,y,z]], device = device)
        end6.record()
        torch.cuda.synchronize()
        vargpu_matrix_trans_times.append(start6.elapsed_time(end6)) 

        start7 = torch.cuda.Event(enable_timing=True)
        end7 = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start7.record()
        f_cuda = torch.tensor([[3.2153548435,1.5484574874,4.1545348574],
                            [3.2153548435,1.5484574874,4.1545348574],
                            [3.2153548435,1.5484574874,4.1545348574],
                            [3.2153548435,1.5484574874,4.1545348574],
                            [3.2153548435,1.5484574874,4.1545348574],
                            [3.2153548435,1.5484574874,4.1545348574]], device = device)
        end7.record()
        torch.cuda.synchronize()
        constgpu_matrix_trans_times.append(start7.elapsed_time(end7))
        """

    logx = True
    logy = True
    n1 = 'addcmul_jit'
    array_times = pd.DataFrame(array_trans_times, columns = [n1])
    array_times[n1] = array_times[ array_times[n1] < array_times[n1].quantile(0.99)]
    array_times.plot.hist(ax = ax, bins = 400, logx = logx, logy=logy, title = 'Operation speed: JIT vs Python\n{} CPU\n {} GPU'.format(cpu,gpu), color = 'r', alpha  = 0.5, xlim = (0.001,1.0))
    print("addcmul_jit Times: points: {}, mean: {}, std: {}, max: {}, min: {}, 25p: {}, 50p: {}, 75p: {}, 90p: {}, 99p: {}".format(len(array_trans_times),
                                                                                                 np.mean(array_trans_times), 
                                                                                                 np.std(array_trans_times), 
                                                                                                 np.max(array_trans_times), 
                                                                                                 np.min(array_trans_times),
                                                                                                 np.percentile(array_trans_times, 25),
                                                                                                 np.percentile(array_trans_times, 50),
                                                                                                 np.percentile(array_trans_times, 75),
                                                                                                 np.percentile(array_trans_times, 90),
                                                                                                 np.percentile(array_trans_times, 99)))

   
    n2 = 'addcmul_py'
    matrix_times = pd.DataFrame(matrix_trans_times, columns = [n2])
    matrix_times[n2] = matrix_times[ matrix_times[n2] < matrix_times[n2].quantile(0.99)]
    matrix_times.plot.hist(ax = ax, bins = 400, logx = logx, logy=logy, color = 'g', alpha  = 0.5)
    print("addcmul_py Times: points: {}, mean: {}, std: {}, max: {}, min: {}".format(len(matrix_trans_times), np.mean(matrix_trans_times), np.std(matrix_trans_times), np.max(matrix_trans_times), np.min(matrix_trans_times)))

    n3 = 'addcmulx5_gpu'
    value_times = pd.DataFrame(value_trans_times, columns = [n3])
    value_times[n3] = value_times[ value_times[n3] < value_times[n3].quantile(0.99)]
    value_times.plot.hist(ax = ax, bins = 400, logx = logx, logy=logy, color = 'b', alpha  = 0.5)
    print("addcmulx5 gpu Times: points: {}, mean: {}, std: {}, max: {}, min: {}".format(len(value_trans_times), np.mean(value_trans_times), np.std(value_trans_times), np.max(value_trans_times), np.min(value_trans_times)))

    n4 = 'addcmulx5_cpu'
    direct_times = pd.DataFrame(direct_trans_times, columns = [n4])
    direct_times[n4] = direct_times[ direct_times[n4] < direct_times[n4].quantile(0.99)]
    direct_times.plot.hist(ax = ax, bins = 400, logx = logx, logy=logy, color = 'm', alpha  = 0.5)
    print("addcmulx5 cpu Times: points: {}, mean: {}, std: {}, max: {}, min: {}".format(len(direct_trans_times), np.mean(direct_trans_times), np.std(direct_trans_times), np.max(direct_trans_times), np.min(direct_trans_times)))

    """
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
    """


    plt.show()



if __name__ == '__main__':
    main()




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