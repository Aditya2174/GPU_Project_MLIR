## Build IREE
```
cd ireeGPU/
cmake -G Ninja -B ../iree-build/ -S .     -DCMAKE_BUILD_TYPE=RelWithDebInfo     -DIREE_ENABLE_ASSERTIONS=ON     -DIREE_ENABLE_SPLIT_DWARF=ON     -DIREE_ENABLE_THIN_ARCHIVES=ON     -DCMAKE_C_COMPILER=clang     -DCMAKE_CXX_COMPILER=clang++     -DIREE_ENABLE_LLD=ON     -DIREE_TARGET_BACKEND_CUDA=ON \
-DIREE_HAL_DRIVER_CUDA=ON

cmake --build ../iree-build/ --target iree-compile iree-benchmark-module iree-run-module -j$(nproc)
```

## Generate the VM Bytecode
```
cd mlir/
../iree-build/tools/iree-compile resnet50_linalg.mlir   --iree-input-type=stablehlo   --iree-hal-target-backends=cuda  --iree-opt-level=O3  -o resnet50_cuda.vmfb
```

## Run the Executible 
```
../iree-build/tools/iree-run-module --device=cuda --module=resnet50_cuda.vmfb --function=forward --input="1x3x224x224xf32=0"
```


### Benchmark the Resnet50 kernel
```
../iree-build/tools/iree-benchmark-module --device=cuda --module=resnet50_cuda.vmfb --function=forward --input="1x3x224x224xf32=0" --batch_size=1 --benchmark_repetitions=100
```


### Obtain the NCU report
```
sudo /usr/local/cuda/bin/ncu -f -o resnet50_report --set full ../iree-build/tools/iree-run-module   --module=resnet50_cuda.vmfb  --device=cuda   --function=forward   --input="1x3x224x224xf32=0"
```
