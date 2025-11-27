## Build IREE
```
cd ireeGPU/
cmake --build . --target iree-compile -j$(nproc)
```

## Generate the VMFB File
~/gpu_project/iree-build/tools/iree-compile resnet50_linalg.mlir   --iree-input-type=stablehlo   --iree-hal-target-backends=cuda  --iree-opt-level=O3  -o resnet50_cuda.vmfb

## Run the Executible 
time ~/gpu_project/iree-build/tools/iree-run-module --device=cuda --module=resnet50_cuda.vmfb --function=forward --input="1x3x224x224xf32=0"


### Benchmark the Resnet50 kernel
~/gpu_project/iree-build/tools/iree-benchmark-module --device=cuda --module=resnet50_cuda.vmfb --function=forward --input="1x3x224x224xf32=0" --batch_size=1 --benchmark_repetitions=100


### Obtain the NCU report
sudo /usr/local/cuda/bin/ncu -f -o resnet50_dyn_tile_report --set full /home/acc1/gpu_project/iree-build/tools/iree-run-module   --module=resnet50_cuda.vmfb  --device=cuda   --function=forward   --input="1x3x224x224xf32=0"
