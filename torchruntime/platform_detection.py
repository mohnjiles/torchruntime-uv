import sys
import platform

from .consts import AMD, INTEL, NVIDIA, CONTACT_LINK

os_name = platform.system()
arch = platform.machine().lower()
py_version = sys.version_info


def get_torch_platform(discrete_gpu_infos):
    """
    Determine the appropriate PyTorch platform to use based on the system architecture, OS, and GPU information.

    Args:
        discrete_gpu_infos (list of tuples): A list where each tuple represents a GPU. Each tuple contains:
            - vendor_id (int): The vendor ID of the GPU (e.g., NVIDIA, AMD, INTEL constants).
            - other details (e.g., model, memory, etc., not used directly in this function).

    Returns:
        str: A string representing the platform to use. Possible values:
            - "cpu": No discrete GPUs or unsupported configuration.
            - "cuXXX": NVIDIA CUDA version (e.g., "cu124").
            - "rocmXXX": AMD ROCm version (e.g., "rocm6.2").
            - "directml": DirectML for AMD or Intel GPUs on Windows.
            - "ipex": Intel Extension for PyTorch for Linux.
            - "xpu": Intel's backend for PyTorch.

    Raises:
        NotImplementedError: For unsupported architectures, OS-GPU combinations, or multiple GPU vendors.
        Warning: Outputs warnings for deprecated Python versions or fallback configurations.
    """

    VALID_ARCHS = {
        "Windows": {"amd64"},
        "Linux": {"x86_64", "aarch64"},
        "Darwin": {"x86_64", "arm64"},
    }

    if arch not in VALID_ARCHS[os_name]:
        raise NotImplementedError(
            f"torch is not currently available for {os_name} on {arch} architecture! If this is no longer true, please contact torch-installer at {CONTACT_LINK}"
        )

    if len(discrete_gpu_infos) == 0:
        return "cpu"

    vendor_ids = set(vendor_id for vendor_id, *_ in discrete_gpu_infos)

    if len(vendor_ids) > 1:
        device_names = list(vendor_name + " " + device_name for _, vendor_name, _, device_name in discrete_gpu_infos)
        raise NotImplementedError(
            f"torch-installer does not currently support multiple graphics card manufacturers on the same computer: {device_names}! Please contact torch-installer at {CONTACT_LINK} with details about your hardware."
        )

    vendor_id = vendor_ids.pop()
    if vendor_id == AMD:
        if os_name == "Windows":
            return "directml"
        elif os_name == "Linux":
            device_names = set(device_name for *_, device_name in discrete_gpu_infos)
            if any(
                device_name.startswith("Navi 3") or device_name.startswith("Navi 2") for device_name in device_names
            ):
                if py_version < (3, 9):
                    print(
                        "[WARNING] Support for Python 3.8 was dropped in ROCm 6.2. torch-installer will default to using ROCm 6.1 instead, but consider switching to a newer Python version to use the latest ROCm!"
                    )
                    return "rocm6.1"
                return "rocm6.2"
            elif any(device_name.startswith("Navi 1") for device_name in device_names):
                return "rocm5.2"
            elif any(device_name.startswith("Ellesmere") for device_name in device_names):
                return "rocm4.2"

            print(
                f"[WARNING] Unsupported AMD graphics card: {device_names}. If this is a recent graphics card (less than 8 years old), please contact torch-installer at {CONTACT_LINK} with details about your hardware."
            )
            return "cpu"
        elif os_name == "Darwin":
            return "rocm4.2"
    elif vendor_id == NVIDIA:
        if os_name in ("Windows", "Linux"):
            return "cu124"
        elif os_name == "Darwin":
            raise NotImplementedError(
                f"torch-installer does not currently support NVIDIA graphics cards on Macs! Please contact torch-installer at {CONTACT_LINK}"
            )
    elif vendor_id == INTEL:
        if os_name == "Windows":
            if py_version < (3, 9):
                print(
                    "[WARNING] Support for Python 3.8 was dropped in torch 2.5, which supports a higher-performance 'xpu' backend for Intel. torch-installer will default to using 'directml' instead, but consider switching to a newer Python version to use the latest 'xpu' backend for Intel!"
                )
                return "directml"
            return "xpu"
        elif os_name == "Linux":
            if py_version < (3, 9):
                print(
                    "[WARNING] Support for Python 3.8 was dropped in torch 2.5, which supports a higher-performance 'xpu' backend for Intel. torch-installer will default to using 'intel-extension-for-pytorch' instead, but consider switching to a newer Python version to use the latest 'xpu' backend for Intel!"
                )
                return "ipex"
            return "xpu"
        else:
            raise NotImplementedError(
                f"torch-installer does not currently support Intel graphics cards on Macs! Please contact torch-installer at {CONTACT_LINK}"
            )

    print(f"Unrecognized vendor: {discrete_gpu_infos}")

    return "cpu"
