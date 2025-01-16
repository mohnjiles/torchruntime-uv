import os
import pytest
from torchruntime.configure import set_rocm_env_vars
from torchruntime.consts import AMD


def create_gpu_info(device_id, device_name):
    return (AMD, "Advanced Micro Devices, Inc. [AMD/ATI]", device_id, device_name)


@pytest.fixture(autouse=True)
def clean_env():
    # Remove relevant environment variables before each test
    env_vars = [
        "HSA_OVERRIDE_GFX_VERSION",
        "HIP_VISIBLE_DEVICES",
        "ROC_ENABLE_PRE_VEGA",
        "HSA_ENABLE_SDMA",
        "FORCE_FULL_PRECISION",
    ]

    # Store original values
    original_values = {}
    for var in env_vars:
        if var in os.environ:
            original_values[var] = os.environ[var]
            del os.environ[var]

    yield

    # Restore original values and remove any new ones
    for var in env_vars:
        if var in os.environ and var not in original_values:
            del os.environ[var]
        elif var in original_values:
            os.environ[var] = original_values[var]


def test_navi_3_settings():
    gpus = [create_gpu_info("123", "Navi 31 XTX")]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ
    assert "FORCE_FULL_PRECISION" not in os.environ


def test_navi_2_settings():
    gpus = [create_gpu_info("123", "Navi 21 XTX")]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ
    assert "FORCE_FULL_PRECISION" not in os.environ


def test_navi_1_settings():
    gpus = [create_gpu_info("123", "Navi 14")]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert os.environ.get("FORCE_FULL_PRECISION") == "yes"
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ


def test_vega_2_settings():
    gpus = [create_gpu_info("123", "Vega 20 Radeon VII")]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "9.0.6"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert "FORCE_FULL_PRECISION" not in os.environ
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ


def test_vega_1_settings():
    gpus = [create_gpu_info("123", "Vega 10")]
    set_rocm_env_vars(gpus, "rocm5.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "9.0.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert "FORCE_FULL_PRECISION" not in os.environ
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ


def test_ellesmere_settings():
    gpus = [create_gpu_info("123", "Ellesmere RX 580")]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "8.0.3"
    assert os.environ.get("ROC_ENABLE_PRE_VEGA") == "1"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert "FORCE_FULL_PRECISION" not in os.environ


def test_unknown_gpu_settings():
    gpus = [create_gpu_info("123", "Unknown GPU")]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert "ROC_ENABLE_PRE_VEGA" not in os.environ
    assert "HIP_VISIBLE_DEVICES" not in os.environ
    assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ
    assert "FORCE_FULL_PRECISION" not in os.environ


def test_multiple_gpus_same_type():
    gpus = [create_gpu_info("123", "Navi 31 XTX"), create_gpu_info("124", "Navi 31 XT")]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0,1"
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ
    assert "FORCE_FULL_PRECISION" not in os.environ


def test_multiple_gpus_mixed_types():
    gpus = [create_gpu_info("123", "Navi 31 XTX"), create_gpu_info("124", "Navi 21 XT")]
    set_rocm_env_vars(gpus, "rocm6.2")

    # Should use Navi 3 settings since at least one GPU is Navi 3
    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0,1"
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ
    assert "FORCE_FULL_PRECISION" not in os.environ


def test_empty_gpu_list():
    gpus = []
    set_rocm_env_vars(gpus, "rocm6.2")

    assert "ROC_ENABLE_PRE_VEGA" not in os.environ
    assert "HIP_VISIBLE_DEVICES" not in os.environ
    assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ
    assert "FORCE_FULL_PRECISION" not in os.environ
