-- Metax MACA GPU support
-- Note: mxcc wrapper at /opt/maca-3.3.0/bin/nvcc needed to filter unsupported flags
--   #!/bin/bash
--   for arg in "$@"; do
--       case "$arg" in -m64|-m32|-W*|--cudart=shared|-gencode|-arch) ;; *) args+=("$arg") ;; esac
--   done
--   exec /opt/maca-3.3.0/mxgpu_llvm/bin/mxcc "${args[@]}"
target("llaisys")
    add_defines("ENABLE_METAX_API")
    add_includedirs("/opt/maca-3.3.0/include/mcr")
    add_linkdirs("/opt/maca-3.3.0/lib")
    add_links("mcblas", "mccl", "mccompiler")
    add_files("../src/device/metax/*.cu")
    add_files("../src/ops/*/metax/*.cu")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end
