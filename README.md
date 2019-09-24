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
Numpy Times: points: 10000, mean: 0.09060891237407923, std: 0.005504961665039473, max: 0.2958720028400421, min: 0.08713600039482117
Matrix Times: points: 10000, mean: 0.028235254419408738, std: 0.002465494683558479, max: 0.1615999937057495, min: 0.016831999644637108
Scalar Times: points: 10000, mean: 0.10172669350206852, std: 0.005380890057001171, max: 0.2979840040206909, min: 0.09011200070381165
Direct Times: points: 10000, mean: 0.09631126805320382, std: 0.00435172602547137, max: 0.2609280049800873, min: 0.08745600283145905
GPUMat Times: points: 10000, mean: 0.025373391924332828, std: 0.0014081497332917897, max: 0.07948800176382065, min: 0.009216000325977802
VGPUMat Times: points: 10000, mean: 0.03115808307286352, std: 0.0015949791753171953, max: 0.10649599879980087, min: 0.009216000325977802
CGPUMat Times: points: 10000, mean: 0.029176448146998882, std: 0.00224004198438089, max: 0.21401600539684296, min: 0.01740800030529499
CUPYMat Times: points: 10000, mean: 0.06909392969030886, std: 0.8441770450322866, max: 84.48172760009766, min: 0.02707199938595295
CPU Pinned Times: points: 10000, mean: 0.027800739184208213, std: 0.002763805926005136, max: 0.21503999829292297, min: 0.026496000587940216
ST CPU Pinned: points: 10000, mean: 0.047590825895499435, std: 0.004946784100365101, max: 0.32521599531173706, min: 0.008736000396311283
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


