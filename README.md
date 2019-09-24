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
SpeedTorch CPU Pinned: points: 10000, mean: 0.04576768335322849, std: 0.0048162757139760546, max: 0.29900801181793213, min: 0.005824000108987093
```

## Jetson Nano Timing
![alt tag](https://github.com/juanmed/torchspeed/blob/master/media/jetson_nano_cpugpu_transfer.png)

```python
Numpy Times: points: 10000, mean: 0.6018594377875328, std: 0.02223985144543474, max: 1.8741140365600586, min: 0.5788019895553589
Matrix Times: points: 10000, mean: 0.18444756867140533, std: 0.016061952086207212, max: 1.4518229961395264, min: 0.17411500215530396
Scalar Times: points: 10000, mean: 0.6519288406074047, std: 0.019685543478514377, max: 1.3126039505004883, min: 0.6256250143051147
Direct Times: points: 10000, mean: 0.637856672745943, std: 0.01853985506062501, max: 1.0831249952316284, min: 0.6150000095367432
GPUMat Times: points: 10000, mean: 0.1736436843469739, std: 0.008710885588477057, max: 0.37421900033950806, min: 0.16145800054073334
VGPUMat Times: points: 10000, mean: 0.19783730645924807, std: 0.0548596756410536, max: 3.488490104675293, min: 0.16145800054073334
CGPUMat Times: points: 10000, mean: 0.19181567124426366, std: 0.03896283223483032, max: 3.035989999771118, min: 0.18088500201702118
CUPYMat Times: points: 10000, mean: 0.3939232293576002, std: 5.595389790043855, max: 559.90087890625, min: 0.3080730140209198
CPU Pinned Times: points: 10000, mean: 0.17686595867425203, std: 0.0083941479989994, max: 0.377811998128891, min: 0.16348999738693237
SpeedTorch CPU Pinned: points: 10000, mean: 0.2667830572873354, std: 0.015449086114074188, max: 0.5225520133972168, min: 0.2483849972486496
```


