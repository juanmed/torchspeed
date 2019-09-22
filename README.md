# torchspeed

From  [here](https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/) and here:

- Minimize the amount of data transferred between host and device when possible, even if that means running kernels on the GPU that get little or no speed-up compared to running them on the host CPU.
- Higher bandwidth is possible between the host and the device when using page-locked (or “pinned”) memory.
- Batching many small transfers into one larger transfer performs much better because it eliminates most of the per-transfer overhead.
- Data transfers between the host and device can sometimes be overlapped with kernel execution and other data transfers.
- Host to GPU copies are much faster when they originate from pinned (page-locked) memory. CPU tensors and storages expose a pin_memory() method, that returns a copy of the object, with data put in a pinned region.
- Also, once you pin a tensor or storage, you can use asynchronous GPU copies. Just pass an additional non_blocking=True argument to a to() or a cuda() call. This can be used to overlap data transfers with computation.
