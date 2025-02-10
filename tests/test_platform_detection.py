import pytest
from torchruntime.device_db import GPU
from torchruntime.platform_detection import get_torch_platform, AMD, NVIDIA, INTEL, os_name, arch, py_version


def test_no_discrete_gpus_windows(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Windows")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "amd64")
    assert get_torch_platform([]) == "cpu"


def test_no_discrete_gpus_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    assert get_torch_platform([]) == "cpu"


def test_no_discrete_gpus_mac(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Darwin")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "arm64")
    assert get_torch_platform([]) == "cpu"


def test_amd_gpu_windows(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Windows")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "amd64")
    gpu_infos = [GPU(AMD, "AMD", 0x1234, "Radeon", True)]
    assert get_torch_platform(gpu_infos) == "directml"


def test_amd_gpu_navi3_linux(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(AMD, "AMD", 0x1234, "Navi 31", True)]
    expected = "rocm6.1" if py_version < (3, 9) else "rocm6.2"
    assert get_torch_platform(gpu_infos) == expected
    if py_version < (3, 9):
        captured = capsys.readouterr()
        assert "Support for Python 3.8 was dropped in ROCm 6.2" in captured.out


def test_amd_gpu_navi2_linux(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(AMD, "AMD", 0x1234, "Navi 22", True)]
    expected = "rocm6.1" if py_version < (3, 9) else "rocm6.2"
    assert get_torch_platform(gpu_infos) == expected
    if py_version < (3, 9):
        captured = capsys.readouterr()
        assert "Support for Python 3.8 was dropped in ROCm 6.2" in captured.out


def test_amd_gpu_navi1_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(AMD, "AMD", 0x1234, "Navi 10", True)]
    assert get_torch_platform(gpu_infos) == "rocm5.2"


def test_amd_gpu_vega2_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(AMD, "AMD", 0x1234, "Vega 20", True)]
    assert get_torch_platform(gpu_infos) == "rocm5.7"


def test_amd_gpu_vega1_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(AMD, "AMD", 0x1234, "Vega 10", True)]
    assert get_torch_platform(gpu_infos) == "rocm5.2"


def test_amd_gpu_ellesmere_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(AMD, "AMD", 0x1234, "Ellesmere", True)]
    assert get_torch_platform(gpu_infos) == "rocm4.2"


def test_amd_gpu_unsupported_linux(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(AMD, "AMD", 0x1234, "UnknownModel", True)]
    assert get_torch_platform(gpu_infos) == "cpu"
    captured = capsys.readouterr()
    assert "[WARNING] Unsupported AMD graphics card" in captured.out


def test_amd_gpu_mac(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Darwin")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "arm64")
    gpu_infos = [GPU(AMD, "AMD", 0x1234, "Radeon", True)]
    assert get_torch_platform(gpu_infos) == "mps"


def test_nvidia_gpu_windows(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Windows")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "amd64")
    gpu_infos = [GPU(NVIDIA, "NVIDIA", 0x1234, "GeForce", True)]
    assert get_torch_platform(gpu_infos) == "cu124"


def test_nvidia_gpu_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(NVIDIA, "NVIDIA", 0x1234, "GeForce", True)]
    assert get_torch_platform(gpu_infos) == "cu124"


def test_nvidia_gpu_mac(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Darwin")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "arm64")
    gpu_infos = [GPU(NVIDIA, "NVIDIA", 0x1234, "GeForce", True)]
    with pytest.raises(NotImplementedError):
        get_torch_platform(gpu_infos)


def test_intel_gpu_windows(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Windows")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "amd64")
    gpu_infos = [GPU(INTEL, "Intel", 0x1234, "Iris", True)]
    expected = "directml" if py_version < (3, 9) else "xpu"
    assert get_torch_platform(gpu_infos) == expected


def test_intel_gpu_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(INTEL, "Intel", 0x1234, "Iris", True)]
    expected = "ipex" if py_version < (3, 9) else "xpu"
    assert get_torch_platform(gpu_infos) == expected


def test_intel_gpu_mac(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Darwin")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "arm64")
    gpu_infos = [GPU(INTEL, "Intel", 0x1234, "Iris", True)]
    with pytest.raises(NotImplementedError):
        get_torch_platform(gpu_infos)


def test_multiple_gpu_vendors_with_NVIDIA(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Windows")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "amd64")
    gpu_infos = [
        GPU(AMD, "AMD", 0x1234, "Radeon", True),
        GPU(NVIDIA, "NVIDIA", 0x5678, "GeForce", True),
    ]
    assert get_torch_platform(gpu_infos) == "cu124"


def test_multiple_gpu_vendors_without_NVIDIA(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Windows")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "amd64")
    gpu_infos = [
        GPU(AMD, "AMD", 0x1234, "Radeon", True),
        GPU(INTEL, "Intel", 0x5678, "Iris", True),
    ]
    with pytest.raises(NotImplementedError):
        get_torch_platform(gpu_infos)


def test_multiple_gpu_NVIDIA(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [
        GPU(NVIDIA, "NVIDIA", "2504", "GA106 [GeForce RTX 3060 Lite Hash Rate]", True),
        GPU(NVIDIA, "NVIDIA", "1c02", "GP106 [GeForce GTX 1060 3GB]", True),
    ]
    assert get_torch_platform(gpu_infos) == "cu124"


def test_multiple_gpu_AMD_Navi3_Navi2(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [
        GPU(AMD, "AMD", "73f0", "Navi 33 [Radeon RX 7600M XT]", True),
        GPU(AMD, "AMD", "73bf", "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]", True),
    ]
    expected = "rocm6.1" if py_version < (3, 9) else "rocm6.2"
    assert get_torch_platform(gpu_infos) == expected
    if py_version < (3, 9):
        captured = capsys.readouterr()
        assert "Support for Python 3.8 was dropped in ROCm 6.2" in captured.out


def test_multiple_gpu_AMD_Navi3_Vega2(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [
        GPU(AMD, "AMD", "73f0", "Navi 33 [Radeon RX 7600M XT]", True),
        GPU(AMD, "AMD", "66af", "Vega 20 [Radeon VII]", True),
    ]
    assert get_torch_platform(gpu_infos) == "rocm5.7"


def test_multiple_gpu_AMD_Vega2_Navi2(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [
        GPU(AMD, "AMD", "66af", "Vega 20 [Radeon VII]", True),
        GPU(AMD, "AMD", "73bf", "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]", True),
    ]
    assert get_torch_platform(gpu_infos) == "rocm5.7"


def test_multiple_gpu_AMD_Vega1_Navi2__incompatible_rocm_version(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [
        GPU(AMD, "AMD", "6867", "Vega 10 XL [Radeon Pro Vega 56]", True),
        GPU(AMD, "AMD", "73bf", "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]", True),
    ]
    expected = "rocm6.1" if py_version < (3, 9) else "rocm6.2"
    assert get_torch_platform(gpu_infos) == expected
    if py_version < (3, 9):
        captured = capsys.readouterr()
        assert "Support for Python 3.8 was dropped in ROCm 6.2" in captured.out
    print("For lack of a better solution at the moment")


def test_multiple_gpu_AMD_Ellesmere_Navi3__incompatible_rocm_version(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [
        GPU(AMD, "AMD", "67df", "Ellesmere [Radeon RX 470/480/570/570X/580/580X/590]", True),
        GPU(AMD, "AMD", "73f0", "Navi 33 [Radeon RX 7600M XT]", True),
    ]
    expected = "rocm6.1" if py_version < (3, 9) else "rocm6.2"
    assert get_torch_platform(gpu_infos) == expected
    if py_version < (3, 9):
        captured = capsys.readouterr()
        assert "Support for Python 3.8 was dropped in ROCm 6.2" in captured.out
    print("For lack of a better solution at the moment")


def test_unsupported_architecture(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "sparc")
    with pytest.raises(NotImplementedError):
        get_torch_platform([])


def test_unrecognized_gpu_vendor(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Windows")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "amd64")
    gpu_infos = [GPU("9999", "UnknownVendor", 0x1234, "Unknown", True)]
    assert get_torch_platform(gpu_infos) == "cpu"


def test_integrated_amd_gfx11_linux(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(AMD, "AMD", "164f", "Phoenix", False)]  # gfx1103
    expected = "rocm6.1" if py_version < (3, 9) else "rocm6.2"
    assert get_torch_platform(gpu_infos) == expected
    if py_version < (3, 9):
        captured = capsys.readouterr()
        assert "Support for Python 3.8 was dropped in ROCm 6.2" in captured.out


def test_integrated_amd_gfx103_linux(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(AMD, "AMD", "163f", "VanGogh", False)]  # gfx1033
    expected = "rocm6.1" if py_version < (3, 9) else "rocm6.2"
    assert get_torch_platform(gpu_infos) == expected
    if py_version < (3, 9):
        captured = capsys.readouterr()
        assert "Support for Python 3.8 was dropped in ROCm 6.2" in captured.out


def test_integrated_amd_gfx90_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(AMD, "AMD", "1636", "Renoir", False)]  # gfx90c
    assert get_torch_platform(gpu_infos) == "rocm5.5"


def test_integrated_amd_unsupported_linux(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(AMD, "AMD", "ffff", "Unknown", False)]  # Not in GPU_DEVICES
    assert get_torch_platform(gpu_infos) == "cpu"
    captured = capsys.readouterr()
    assert "[WARNING] Unsupported AMD APU" in captured.out


def test_integrated_amd_windows(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Windows")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "amd64")
    gpu_infos = [GPU(AMD, "AMD", "164f", "Phoenix", False)]
    assert get_torch_platform(gpu_infos) == "directml"


def test_integrated_amd_mac(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Darwin")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "arm64")
    gpu_infos = [GPU(AMD, "AMD", "164f", "Phoenix", False)]
    assert get_torch_platform(gpu_infos) == "cpu"
    captured = capsys.readouterr()
    assert "torchruntime does not currently support integrated graphics cards on Macs" in captured.out


def test_integrated_intel_linux(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [GPU(INTEL, "Intel", "0x56c1", "UHD Graphics", False)]
    expected = "ipex" if py_version < (3, 9) else "xpu"
    assert get_torch_platform(gpu_infos) == expected
    if py_version < (3, 9):
        captured = capsys.readouterr()
        assert "Support for Python 3.8 was dropped in torch 2.5" in captured.out


def test_integrated_intel_windows(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Windows")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "amd64")
    gpu_infos = [GPU(INTEL, "Intel", "0x56c1", "UHD Graphics", False)]
    assert get_torch_platform(gpu_infos) == "directml"


def test_integrated_intel_mac(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Darwin")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "arm64")
    gpu_infos = [GPU(INTEL, "Intel", "0x56c1", "UHD Graphics", False)]
    assert get_torch_platform(gpu_infos) == "cpu"
    captured = capsys.readouterr()
    assert "torchruntime does not currently support integrated graphics cards on Macs" in captured.out


def test_mixed_amd_discrete_integrated_linux(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [
        GPU(AMD, "AMD", "164f", "Phoenix", False),  # integrated
        GPU(AMD, "AMD", "73f0", "Navi 33", True),  # discrete
    ]
    expected = "rocm6.1" if py_version < (3, 9) else "rocm6.2"
    assert get_torch_platform(gpu_infos) == expected
    if py_version < (3, 9):
        captured = capsys.readouterr()
        assert "Support for Python 3.8 was dropped in ROCm 6.2" in captured.out


def test_mixed_nvidia_discrete_amd_integrated_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [
        GPU(AMD, "AMD", "164f", "Phoenix", False),  # integrated
        GPU(NVIDIA, "NVIDIA", "2504", "RTX 3060", True),  # discrete
    ]
    assert get_torch_platform(gpu_infos) == "cu124"


def test_mixed_nvidia_discrete_intel_integrated_windows(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Windows")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "amd64")
    gpu_infos = [
        GPU(INTEL, "Intel", "0x56c1", "UHD Graphics", False),  # integrated
        GPU(NVIDIA, "NVIDIA", "2504", "RTX 3060", True),  # discrete
    ]
    assert get_torch_platform(gpu_infos) == "cu124"


def test_mixed_amd_discrete_intel_integrated_linux(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [
        GPU(INTEL, "Intel", "0x56c1", "UHD Graphics", False),  # integrated
        GPU(AMD, "AMD", "73f0", "Navi 33", True),  # discrete
    ]
    expected = "rocm6.1" if py_version < (3, 9) else "rocm6.2"
    assert get_torch_platform(gpu_infos) == expected
    if py_version < (3, 9):
        captured = capsys.readouterr()
        assert "Support for Python 3.8 was dropped in ROCm 6.2" in captured.out


def test_mixed_multiple_discrete_and_integrated(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    gpu_infos = [
        GPU(AMD, "AMD", "164f", "Phoenix", False),  # integrated AMD
        GPU(NVIDIA, "NVIDIA", "2504", "RTX 3060", True),  # discrete NVIDIA
        GPU(AMD, "AMD", "73f0", "Navi 33", True),  # discrete AMD
    ]
    assert get_torch_platform(gpu_infos) == "cu124"
