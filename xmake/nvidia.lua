-- Add CUDA source files directly to the shared library target
-- so nvcc handles all CUDA linking correctly
target("llaisys")
    add_files("../src/device/nvidia/*.cu")
    add_files("../src/ops/*/nvidia/*.cu")
    add_links("cudart", "cudadevrt")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end
