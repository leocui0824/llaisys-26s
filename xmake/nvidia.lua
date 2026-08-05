target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_links("cudart")
    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

-- Add nvidia device runtime to device target
target("llaisys-device")
    add_deps("llaisys-device-nvidia")

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_links("cudart")
    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()

-- Add nvidia ops to ops target
target("llaisys-ops")
    add_deps("llaisys-ops-nvidia")
