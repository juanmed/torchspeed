# torchspeed

From  [here](https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/) and [here](https://pytorch.org/docs/master/notes/cuda.html):

- Minimize the amount of data transferred between host and device when possible, even if that means running kernels on the GPU that get little or no speed-up compared to running them on the host CPU.
- Higher bandwidth is possible between the host and the device when using page-locked (or “pinned”) memory.
- Batching many small transfers into one larger transfer performs much better because it eliminates most of the per-transfer overhead.
- Data transfers between the host and device can sometimes be overlapped with kernel execution and other data transfers.
- Host to GPU copies are much faster when they originate from pinned (page-locked) memory. CPU tensors and storages expose a pin_memory() method, that returns a copy of the object, with data put in a pinned region.
- Also, once you pin a tensor or storage, you can use asynchronous GPU copies. Just pass an additional non_blocking=True argument to a to() or a cuda() call. This can be used to overlap data transfers with computation.


![alt tag](https://github.com/juanmed/torchspeed/blob/master/media/desktop_cpugpu_transfer.png)

```python 
Numpy Times: points: 10000, mean: 0.08801872953251004, std: 0.004730710669616504, max: 0.2807680070400238, min: 0.07625599950551987
Matrix Times: points: 10000, mean: 0.02691661435170099, std: 0.002060475930085663, max: 0.1695999950170517, min: 0.015552000142633915
Scalar Times: points: 10000, mean: 0.09977850218713284, std: 0.004403828310264877, max: 0.25705599784851074, min: 0.08601599931716919
Direct Times: points: 10000, mean: 0.09558431115597486, std: 0.006109912391678707, max: 0.28748801350593567, min: 0.0870399996638298
GPUMat Times: points: 10000, mean: 0.02490384978670627, std: 0.0015759606943080674, max: 0.08089599758386612, min: 0.005119999870657921
VGPUMat Times: points: 10000, mean: 0.030266073651798068, std: 0.0015619499290569344, max: 0.09523200243711472, min: 0.005119999870657921
CGPUMat Times: points: 10000, mean: 0.02866346567682922, std: 0.0035306552234948144, max: 0.24022400379180908, min: 0.017855999991297722
CUPYMat Times: points: 10000, mean: 0.06792415700107812, std: 0.8546480585564749, max: 85.52751922607422, min: 0.02502400055527687
CPU Pinned Times: points: 20000, mean: 0.02516704963631928, std: 0.0032678948035097413, max: 0.20479999482631683, min: 0.021503999829292297
ST CPU Pinned: points: 10000, mean: 0.04576768335322849, std: 0.0048162757139760546, max: 0.29900801181793213, min: 0.005824000108987093
```

