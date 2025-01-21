import os

from .device_db import get_discrete_gpus
from .platform_detection import get_torch_platform, os_name


def init_torch():
    discrete_gpu_infos = get_discrete_gpus()
    torch_platform = get_torch_platform(discrete_gpu_infos)

    if torch_platform.startswith("rocm"):
        check_rocm_permissions()
        set_rocm_env_vars(discrete_gpu_infos, torch_platform)
    elif os_name == "Darwin":
        _set_env_vars({"PYTORCH_ENABLE_MPS_FALLBACK": "1"})


def check_rocm_permissions():
    if not os.access("/dev/kfd", os.W_OK):
        print(
            """#########################################################################
            #                    No write access to /dev/kfd !                      #
            #########################################################################

            Without this, the ROCm driver will probably not be able to initialize the GPU and torch will use the CPU for rendering.

            Follow the instructions on this site to configure access to /dev/kfd:
            https://github.com/easydiffusion/easydiffusion/wiki/AMD-on-Linux#access-permissions"""
        )


def set_rocm_env_vars(discrete_gpu_infos, torch_platform):
    if not discrete_gpu_infos:
        return

    device_names = [device_name for *_, device_name in discrete_gpu_infos]
    env = {}

    # interesting reading:
    # gfx config from: https://web.archive.org/web/20241228163540/https://llvm.org/docs/AMDGPUUsage.html#processors
    # more info: https://web.archive.org/web/20241209013717/https://discuss.linuxcontainers.org/t/rocm-and-pytorch-on-amd-apu-or-gpu-ai/19743
    # this thread is great for understanding the status of torch support for RDNA 1 (i.e. 5000 series): https://github.com/ROCm/ROCm/issues/2527
    # past settings from: https://github.com/easydiffusion/easydiffusion/blob/20d77a85a1ed766ece0cc4b6a55dca003bce262c/scripts/check_modules.py#L405-L420

    # Determine GPU generations present
    has_navi3 = any("Navi 3" in device_name for device_name in device_names)  # RX 7000 series
    has_navi2 = any("Navi 2" in device_name for device_name in device_names)  # RX 6000 series
    has_navi1 = any("Navi 1" in device_name for device_name in device_names)  # RX 5000 series
    has_vega2 = any("Vega 2" in device_name for device_name in device_names)  # Radeon VII etc
    has_vega1 = any("Vega 1" in device_name for device_name in device_names)  # Radeon RX Vega 56 etc
    has_ellesmere = any("Ellesmere" in device_name for device_name in device_names)  # RX 570/580/Polaris etc

    # Select GPU generation settings based on priority
    if has_navi3:
        env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
        # Find the index of the first Navi 3 GPU
        env["HIP_VISIBLE_DEVICES"] = _visible_device_ids(discrete_gpu_infos, "Navi 3")
    elif has_navi2:
        env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        # Find the index of the first Navi 2 GPU
        env["HIP_VISIBLE_DEVICES"] = _visible_device_ids(discrete_gpu_infos, "Navi 2")
    elif has_navi1:
        env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        # env["HSA_ENABLE_SDMA"] = "0"  # uncomment this if facing errors like in https://github.com/ROCm/ROCm/issues/2616
        env["FORCE_FULL_PRECISION"] = "yes"  # https://github.com/ROCm/ROCm/issues/2527
        # FORCE_FULL_PRECISION won't be necessary once this is fixed (and torch2 wheels are released for ROCm 6.2): https://github.com/pytorch/pytorch/issues/132570#issuecomment-2313071756
        env["HIP_VISIBLE_DEVICES"] = _visible_device_ids(discrete_gpu_infos, "Navi 1")
    elif has_vega2:
        env["HSA_OVERRIDE_GFX_VERSION"] = "9.0.6"
        env["HIP_VISIBLE_DEVICES"] = _visible_device_ids(discrete_gpu_infos, "Vega 2")
    elif has_vega1:
        # # https://discord.com/channels/1014774730907209781/1329021732794667068/1329261488300363776
        env["HSA_OVERRIDE_GFX_VERSION"] = "9.0.0"
        env["HIP_VISIBLE_DEVICES"] = _visible_device_ids(discrete_gpu_infos, "Vega 1")
    elif has_ellesmere:
        env["HSA_OVERRIDE_GFX_VERSION"] = "8.0.3"  # https://github.com/ROCm/ROCm/issues/1659
        env["ROC_ENABLE_PRE_VEGA"] = "1"
        env["HIP_VISIBLE_DEVICES"] = _visible_device_ids(discrete_gpu_infos, "Ellesmere")
    else:
        env["ROC_ENABLE_PRE_VEGA"] = "1"
        print(f"[WARNING] Unrecognized AMD graphics card: {device_names}")
        return

    _set_env_vars(env)


def _visible_device_ids(discrete_gpu_info, family_name):
    ids = [str(i) for i, (*_, name) in enumerate(discrete_gpu_info) if family_name in name]
    return ",".join(ids)


def _set_env_vars(env):
    for k, v in env.items():
        print(f"[INFO] Setting env variable {k}={v}")
        os.environ[k] = v
