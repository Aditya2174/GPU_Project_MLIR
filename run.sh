# build iree
cd ireeGPU/
cmake --build ../iree-build/ --target iree-compile iree-benchmark-module iree-run-module -j$(nproc)

cd ../mlir/
# generate the VM Bytecode
../iree-build/tools/iree-compile resnet50_linalg.mlir   --iree-input-type=stablehlo   --iree-hal-target-backends=cuda  --iree-opt-level=O3  -o resnet50_cuda.vmfb

# benchmarking 
../iree-build/tools/iree-benchmark-module --device=cuda --module=resnet50_cuda.vmfb --function=forward --input="1x3x224x224xf32=0" --batch_size=1 --benchmark_repetitions=100