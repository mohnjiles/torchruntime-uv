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


def test_rocm_navi_3_settings():
    gpus = [create_gpu_info("123", "Navi 31 XTX")]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ
    assert "FORCE_FULL_PRECISION" not in os.environ


def test_rocm_navi_2_settings():
    gpus = [create_gpu_info("123", "Navi 21 XTX")]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ
    assert "FORCE_FULL_PRECISION" not in os.environ


def test_rocm_navi_1_settings():
    gpus = [create_gpu_info("123", "Navi 14")]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert os.environ.get("FORCE_FULL_PRECISION") == "yes"
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ


def test_rocm_vega_2_settings():
    gpus = [create_gpu_info("123", "Vega 20 Radeon VII")]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "9.0.6"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert "FORCE_FULL_PRECISION" not in os.environ
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ


def test_rocm_vega_1_settings():
    gpus = [create_gpu_info("123", "Vega 10")]
    set_rocm_env_vars(gpus, "rocm5.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "9.0.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert "FORCE_FULL_PRECISION" not in os.environ
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ


def test_rocm_ellesmere_settings():
    gpus = [create_gpu_info("123", "Ellesmere RX 580")]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "8.0.3"
    assert os.environ.get("ROC_ENABLE_PRE_VEGA") == "1"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert "FORCE_FULL_PRECISION" not in os.environ


def test_rocm_unknown_gpu_settings():
    gpus = [create_gpu_info("123", "Unknown GPU")]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert "ROC_ENABLE_PRE_VEGA" not in os.environ
    assert "HIP_VISIBLE_DEVICES" not in os.environ
    assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ
    assert "FORCE_FULL_PRECISION" not in os.environ


def test_rocm_multiple_gpus_same_model():
    gpus = [create_gpu_info("123", "Navi 31 XTX"), create_gpu_info("124", "Navi 31 XT")]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0,1"
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ
    assert "FORCE_FULL_PRECISION" not in os.environ


def print_gpu_wasted_warning():
    print(
        "Fixme: This is not ideal, because we're disabling a perfectly-usable GPU. Need a way to specify which GPU to use (in separate processes), instead of trying to run both in the same python process"
    )


def test_rocm_multiple_gpus_navi3_navi2__newer_gpu_first():
    gpus = [
        create_gpu_info("73f0", "Navi 33 [Radeon RX 7600M XT]"),
        create_gpu_info("73bf", "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]"),
    ]
    set_rocm_env_vars(gpus, "rocm6.2")

    # Should use Navi 3 settings since at least one GPU is Navi 3
    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ
    assert "FORCE_FULL_PRECISION" not in os.environ

    print_gpu_wasted_warning()


def test_rocm_multiple_gpus_navi2_navi3__newer_gpu_second():
    gpus = [
        create_gpu_info("73bf", "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]"),
        create_gpu_info("73f0", "Navi 33 [Radeon RX 7600M XT]"),
    ]
    set_rocm_env_vars(gpus, "rocm6.2")

    # Should use Navi 3 settings since at least one GPU is Navi 3
    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "1"
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ
    assert "FORCE_FULL_PRECISION" not in os.environ

    print_gpu_wasted_warning()


def test_rocm_multiple_gpus_vega2_navi2():
    gpus = [
        create_gpu_info("66af", "Vega 20 [Radeon VII]"),
        create_gpu_info("73bf", "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]"),
    ]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "1"
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ
    assert "FORCE_FULL_PRECISION" not in os.environ

    print_gpu_wasted_warning()


def test_rocm_multiple_gpus_navi2_vega1():
    gpus = [
        create_gpu_info("73bf", "Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]"),
        create_gpu_info("6867", "Vega 10 XL [Radeon Pro Vega 56]"),
    ]
    set_rocm_env_vars(gpus, "rocm6.2")

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert "FORCE_FULL_PRECISION" not in os.environ
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ

    print_gpu_wasted_warning()


def test_rocm_multiple_gpus_navi3_ellesmere():
    gpus = [
        create_gpu_info("73f0", "Navi 33 [Radeon RX 7600M XT]"),
        create_gpu_info("67df", "Ellesmere [Radeon RX 470/480/570/570X/580/580X/590]"),
    ]
    set_rocm_env_vars(gpus, "rocm6.2")  # need to figure this out

    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == "11.0.0"
    assert os.environ.get("HIP_VISIBLE_DEVICES") == "0"
    assert "ROC_ENABLE_PRE_VEGA" not in os.environ
    assert "FORCE_FULL_PRECISION" not in os.environ

    print_gpu_wasted_warning()


def test_rocm_empty_gpu_list():
    gpus = []
    set_rocm_env_vars(gpus, "rocm6.2")

    assert "ROC_ENABLE_PRE_VEGA" not in os.environ
    assert "HIP_VISIBLE_DEVICES" not in os.environ
    assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ
    assert "FORCE_FULL_PRECISION" not in os.environ
