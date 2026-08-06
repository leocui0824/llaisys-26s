-- Metax MACA GPU support
target("llaisys")
    add_defines("ENABLE_METAX_API")
    add_includedirs("/opt/maca-3.3.0/include/mcr")
    add_linkdirs("/opt/maca-3.3.0/lib")
    add_links("mcrt", "mclibxt")
    add_files("../src/device/metax/*.cu")
    add_files("../src/ops/*/metax/*.cu")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end
