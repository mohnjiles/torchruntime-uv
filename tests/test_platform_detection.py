import pytest
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
    discrete_gpu_infos = [(AMD, "AMD", 0x1234, "Radeon")]
    assert get_torch_platform(discrete_gpu_infos) == "directml"


def test_amd_gpu_navi3_linux(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    discrete_gpu_infos = [(AMD, "AMD", 0x1234, "Navi 31")]
    expected = "rocm6.1" if py_version < (3, 9) else "rocm6.2"
    assert get_torch_platform(discrete_gpu_infos) == expected
    if py_version < (3, 9):
        captured = capsys.readouterr()
        assert "Support for Python 3.8 was dropped in ROCm 6.2" in captured.out


def test_amd_gpu_navi2_linux(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    discrete_gpu_infos = [(AMD, "AMD", 0x1234, "Navi 22")]
    expected = "rocm6.1" if py_version < (3, 9) else "rocm6.2"
    assert get_torch_platform(discrete_gpu_infos) == expected
    if py_version < (3, 9):
        captured = capsys.readouterr()
        assert "Support for Python 3.8 was dropped in ROCm 6.2" in captured.out


def test_amd_gpu_navi1_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    discrete_gpu_infos = [(AMD, "AMD", 0x1234, "Navi 10")]
    assert get_torch_platform(discrete_gpu_infos) == "rocm5.2"


def test_amd_gpu_vega2_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    discrete_gpu_infos = [(AMD, "AMD", 0x1234, "Vega 20")]
    assert get_torch_platform(discrete_gpu_infos) == "rocm5.7"


def test_amd_gpu_vega1_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    discrete_gpu_infos = [(AMD, "AMD", 0x1234, "Vega 10")]
    assert get_torch_platform(discrete_gpu_infos) == "rocm5.2"


def test_amd_gpu_ellesmere_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    discrete_gpu_infos = [(AMD, "AMD", 0x1234, "Ellesmere")]
    assert get_torch_platform(discrete_gpu_infos) == "rocm4.2"


def test_amd_gpu_unsupported_linux(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    discrete_gpu_infos = [(AMD, "AMD", 0x1234, "UnknownModel")]
    assert get_torch_platform(discrete_gpu_infos) == "cpu"
    captured = capsys.readouterr()
    assert "[WARNING] Unsupported AMD graphics card" in captured.out


def test_amd_gpu_mac(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Darwin")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "arm64")
    discrete_gpu_infos = [(AMD, "AMD", 0x1234, "Radeon")]
    assert get_torch_platform(discrete_gpu_infos) == "mps"


def test_nvidia_gpu_windows(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Windows")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "amd64")
    discrete_gpu_infos = [(NVIDIA, "NVIDIA", 0x1234, "GeForce")]
    assert get_torch_platform(discrete_gpu_infos) == "cu124"


def test_nvidia_gpu_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    discrete_gpu_infos = [(NVIDIA, "NVIDIA", 0x1234, "GeForce")]
    assert get_torch_platform(discrete_gpu_infos) == "cu124"


def test_nvidia_gpu_mac(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Darwin")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "arm64")
    discrete_gpu_infos = [(NVIDIA, "NVIDIA", 0x1234, "GeForce")]
    with pytest.raises(NotImplementedError):
        get_torch_platform(discrete_gpu_infos)


def test_intel_gpu_windows(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Windows")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "amd64")
    discrete_gpu_infos = [(INTEL, "Intel", 0x1234, "Iris")]
    expected = "directml" if py_version < (3, 9) else "xpu"
    assert get_torch_platform(discrete_gpu_infos) == expected


def test_intel_gpu_linux(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    discrete_gpu_infos = [(INTEL, "Intel", 0x1234, "Iris")]
    expected = "ipex" if py_version < (3, 9) else "xpu"
    assert get_torch_platform(discrete_gpu_infos) == expected


def test_intel_gpu_mac(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Darwin")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "arm64")
    discrete_gpu_infos = [(INTEL, "Intel", 0x1234, "Iris")]
    with pytest.raises(NotImplementedError):
        get_torch_platform(discrete_gpu_infos)


def test_multiple_gpu_vendors(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Windows")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "amd64")
    discrete_gpu_infos = [
        (AMD, "AMD", 0x1234, "Radeon"),
        (NVIDIA, "NVIDIA", 0x5678, "GeForce"),
    ]
    with pytest.raises(NotImplementedError):
        get_torch_platform(discrete_gpu_infos)


def test_multiple_gpu_NVIDIA(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    discrete_gpu_infos = [
        (NVIDIA, "NVIDIA", "2504", "GA106 [GeForce RTX 3060 Lite Hash Rate]"),
        (NVIDIA, "NVIDIA", "1c02", "GP106 [GeForce GTX 1060 3GB]"),
    ]
    assert get_torch_platform(discrete_gpu_infos) == "cu124"


def test_multiple_gpu_AMD_Navi3_Navi2(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    discrete_gpu_infos = [
        (AMD, "AMD", "73f0", "Navi 33 [Radeon RX 7600M XT]"),
        (AMD, "AMD", "73bf", "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]"),
    ]
    expected = "rocm6.1" if py_version < (3, 9) else "rocm6.2"
    assert get_torch_platform(discrete_gpu_infos) == expected
    if py_version < (3, 9):
        captured = capsys.readouterr()
        assert "Support for Python 3.8 was dropped in ROCm 6.2" in captured.out


def test_multiple_gpu_AMD_Navi3_Vega2(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    discrete_gpu_infos = [
        (AMD, "AMD", "73f0", "Navi 33 [Radeon RX 7600M XT]"),
        (AMD, "AMD", "66af", "Vega 20 [Radeon VII]"),
    ]
    assert get_torch_platform(discrete_gpu_infos) == "rocm5.7"


def test_multiple_gpu_AMD_Vega2_Navi2(monkeypatch):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    discrete_gpu_infos = [
        (AMD, "AMD", "66af", "Vega 20 [Radeon VII]"),
        (AMD, "AMD", "73bf", "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]"),
    ]
    assert get_torch_platform(discrete_gpu_infos) == "rocm5.7"


def test_multiple_gpu_AMD_Vega1_Navi2__incompatible_rocm_version(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    discrete_gpu_infos = [
        (AMD, "AMD", "6867", "Vega 10 XL [Radeon Pro Vega 56]"),
        (AMD, "AMD", "73bf", "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]"),
    ]
    expected = "rocm6.1" if py_version < (3, 9) else "rocm6.2"
    assert get_torch_platform(discrete_gpu_infos) == expected
    if py_version < (3, 9):
        captured = capsys.readouterr()
        assert "Support for Python 3.8 was dropped in ROCm 6.2" in captured.out
    print("For lack of a better solution at the moment")


def test_multiple_gpu_AMD_Ellesmere_Navi3__incompatible_rocm_version(monkeypatch, capsys):
    monkeypatch.setattr("torchruntime.platform_detection.os_name", "Linux")
    monkeypatch.setattr("torchruntime.platform_detection.arch", "x86_64")
    discrete_gpu_infos = [
        (AMD, "AMD", "67df", "Ellesmere [Radeon RX 470/480/570/570X/580/580X/590]"),
        (AMD, "AMD", "73f0", "Navi 33 [Radeon RX 7600M XT]"),
    ]
    expected = "rocm6.1" if py_version < (3, 9) else "rocm6.2"
    assert get_torch_platform(discrete_gpu_infos) == expected
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
    discrete_gpu_infos = [("9999", "UnknownVendor", 0x1234, "Unknown")]
    assert get_torch_platform(discrete_gpu_infos) == "cpu"
