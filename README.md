# torchspeed

From  [here](https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/) and [here](https://pytorch.org/docs/master/notes/cuda.html):

- Minimize the amount of data transferred between host and device when possible, even if that means running kernels on the GPU that get little or no speed-up compared to running them on the host CPU.
- Higher bandwidth is possible between the host and the device when using page-locked (or “pinned”) memory.
- Batching many small transfers into one larger transfer performs much better because it eliminates most of the per-transfer overhead.
- Data transfers between the host and device can sometimes be overlapped with kernel execution and other data transfers.
- Host to GPU copies are much faster when they originate from pinned (page-locked) memory. CPU tensors and storages expose a pin_memory() method, that returns a copy of the object, with data put in a pinned region.
- Also, once you pin a tensor or storage, you can use asynchronous GPU copies. Just pass an additional non_blocking=True argument to a to() or a cuda() call. This can be used to overlap data transfers with computation.

## Desktop Timing
![alt tag](https://github.com/juanmed/torchspeed/blob/master/media/desktop_cpugpu_transfer.png)


## Jetson Nano Timing
![alt tag](https://github.com/juanmed/torchspeed/blob/master/media/jetson_nano_cpugpu_transfer.png)

```python
Numpy Times: points: 10000, mean: 0.6005140175342559, std: 0.01822966386692426, max: 1.029062032699585, min: 0.570937991142273
Matrix Times: points: 10000, mean: 0.18131085225343704, std: 0.015875195807204902, max: 1.5151560306549072, min: 0.1709890067577362
Scalar Times: points: 10000, mean: 0.6479232557177543, std: 0.019200663093293647, max: 1.1922399997711182, min: 0.6150519847869873
Direct Times: points: 10000, mean: 0.631741645860672, std: 0.02557701268243619, max: 1.9954169988632202, min: 0.6038020253181458
GPUMat Times: points: 10000, mean: 0.17211523707956075, std: 0.009304733290348713, max: 0.38630199432373047, min: 0.160521000623703
VGPUMat Times: points: 10000, mean: 0.19833447985202074, std: 0.05192984135644705, max: 2.9525530338287354, min: 0.160521000623703
CGPUMat Times: points: 10000, mean: 0.1914666683986783, std: 0.03367670842509451, max: 3.4513540267944336, min: 0.1816670000553131
CUPYMat Times: points: 10000, mean: 0.3991811771035194, std: 5.688358450540875, max: 569.1990966796875, min: 0.31083399057388306
CPU Pinned Times: points: 20000, mean: 0.1694693487741053, std: 0.011078960266568207, max: 0.3476560115814209, min: 0.1510930061340332
ST CPU Pinned: points: 10000, mean: 0.26537555684298275, std: 0.015760937672519612, max: 0.5634899735450745, min: 0.24500000476837158
```

