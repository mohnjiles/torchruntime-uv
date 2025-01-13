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
    device_names = [device_name for *_, device_name in discrete_gpu_infos]

    env = {}

    # interesting reading:
    # gfx config from: https://web.archive.org/web/20241228163540/https://llvm.org/docs/AMDGPUUsage.html#processors
    # more info: https://web.archive.org/web/20241209013717/https://discuss.linuxcontainers.org/t/rocm-and-pytorch-on-amd-apu-or-gpu-ai/19743
    # this thread is great for understanding the status of torch support for RDNA 1 (i.e. 5000 series): https://github.com/ROCm/ROCm/issues/2527
    # past settings from: https://github.com/easydiffusion/easydiffusion/blob/20d77a85a1ed766ece0cc4b6a55dca003bce262c/scripts/check_modules.py#L405-L420

    if any(device_name.startswith("Navi 3") for device_name in device_names):
        print("[INFO] Applying Navi 3x settings")
        env["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
    elif any(device_name.startswith("Navi 2") for device_name in device_names):
        print("[INFO] Applying Navi 2x settings")
        env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    elif any(device_name.startswith("Navi 1") for device_name in device_names):
        print("[INFO] Applying Navi 1x settings")
        env["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        # env["HSA_ENABLE_SDMA"] = "0"  # uncomment this if facing errors like in https://github.com/ROCm/ROCm/issues/2616
        env["FORCE_FULL_PRECISION"] = "yes"  # https://github.com/ROCm/ROCm/issues/2527
        # FORCE_FULL_PRECISION won't be necessary once this is fixed (and torch2 wheels are released for ROCm 6.2): https://github.com/pytorch/pytorch/issues/132570#issuecomment-2313071756
    elif any(device_name.startswith("Ellesmere") for device_name in device_names):  # RX 570, 580, 590, Polaris etc
        print("[INFO] Applying Ellesmere settings")
        env["HSA_OVERRIDE_GFX_VERSION"] = "8.0.3"  # https://github.com/ROCm/ROCm/issues/1659
        env["ROC_ENABLE_PRE_VEGA"] = "1"
    else:
        env["ROC_ENABLE_PRE_VEGA"] = "1"
        print(f"[WARNING] Unrecognized AMD graphics card: {device_names}")
        return

    num_devices = len(device_names)
    env["HIP_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_devices))

    _set_env_vars(env)


def _set_env_vars(env):
    for k, v in env.items():
        print(f"[INFO] Setting env variable {k}={v}")
        os.environ[k] = v
