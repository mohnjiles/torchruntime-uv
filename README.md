# torchruntime
[![Discord Server](https://img.shields.io/discord/1014774730907209781?label=Discord)](https://discord.com/invite/u9yhsFmEkB)

**torchruntime** is a lightweight package for automatically installing the appropriate variant of PyTorch on a user's computer, based on their OS, and GPU manufacturer and GPU model.

This package is used by [Easy Diffusion](https://github.com/easydiffusion/easydiffusion), but you're welcome to use it as well. It's useful for developers who make PyTorch-based apps that target users with NVIDIA, AMD and Intel graphics cards (as well as CPU-only usage), on Windows, Mac and Linux.

### Why?
It lets you treat PyTorch as a single dependency (like it should be), and lets you assume that each user will get the most-performant variant of PyTorch suitable for their computer's OS and hardware.

It deals with the complexity of the variety of torch builds and configurations required for CUDA, AMD (ROCm, DirectML), Intel (xpu/DirectML/ipex), and CPU-only.

**Compatibility table**: [Click here](#compatibility-table) to see the supported graphics cards and operating systems.

# Installation
Supports Windows, Linux, and Mac.

`pip install torchruntime`

## Usage
### Step 1. Install the appropriate variant of PyTorch
*This command should be run on the user's computer, or while creating platform-specific builds:*

`python -m torchruntime install`

This will install `torch`, `torchvision`, and `torchaudio`, and will decide the variant based on the user's OS, GPU manufacturer and GPU model number. See [customizing packages](#customizing-packages) for more options.

### Step 2. Initialize torch
This should be run inside your program, to initialize the required environment variables (if any) for the variant of torch being used.

```py
import torchruntime

torchruntime.init_torch()
```

## Customizing packages
By default, `python -m torchruntime install` will install the latest available `torch`, `torchvision` and `torchaudio` suitable on the user's platform.

You can customize the packages to install by including their names:
* For e.g. to install only `torch` and `torchvision`, you can run `python -m torchruntime install torch torchvision`
* To install specific versions (in pip format), you can run `python -m torchruntime install "torch>2.0" "torchvision==0.20"`

**Note:** If you specify package versions, please keep in mind that the version may not be available to *all* the users on *all* the torch platforms. For e.g. a user with Python 3.8 would not be able to install torch 2.5 (or higher), because torch 2.5 dropped support for Python 3.8.

So in general, it's better to avoid specifying a version unless it really matters to you (or you know what you're doing). Instead, please allow `torchruntime` to pick the latest-possible version for the user.

# Compatibility table
The list of platforms on which `torchruntime` can install a working variant of PyTorch.

**Note:** *This list is based on user feedback (since I don't have all the cards). Please let me know if your card is supported (or not) by opening a pull request or issue or messaging on [Discord](https://discord.com/invite/u9yhsFmEkB) (with supporting logs).*

**CPU-only:**

| OS  | Supported?| Notes  |
|---|---|---|
| Windows  | ✅ Yes  | x86_64  |
| Linux  | ✅ Yes  | x86_64 and aarch64  |
| Mac (M1/M2/M3/M4)  | ✅ Yes  | arm64. `mps` backend  |
| Mac (Intel)  | ✅ Yes  | x86_64. Stopped after `torch 2.2.2`  |

**NVIDIA:**

| Series  | Supported? | OS | Notes  |
|---|---|---|---|
| 40xx  | ✅ Yes  | Win/Linux  | Uses CUDA 124  |
| 30xx  | ✅ Yes  | Win/Linux  | Uses CUDA 124  |
| 20xx  | ✅ Yes  | Win/Linux  | Uses CUDA 124  |
| 10xx/16xx  | ✅ Yes  | Win/Linux  | Uses CUDA 124. Full-precision required on 16xx series  |

**AMD:**

| Series  | Supported? | OS   | Notes  |
|---|---|---|---|
| 7xxx  | ✅ Yes  | Win/Linux    | Navi3/RDNA3 (gfx110x). ROCm 6.2 on Linux. DirectML on Windows  |
| 6xxx  | ✅ Yes  | Win/Linux    | Navi2/RDNA2 (gfx103x). ROCm 6.2 on Linux. DirectML on Windows  |
| 5xxx  | ✅ Yes  | Win/Linux    | Navi1/RDNA1 (gfx101x). Full-precision required. DirectML on Windows. Linux only supports upto ROCm 5.2. Waiting for [this](https://github.com/pytorch/pytorch/issues/132570#issuecomment-2313071756) for ROCm 6.2 support.  |
| 5xxx on Intel Mac  | ❓ Untested (WIP)  | Intel Mac  | gfx101x. Implemented but need testers, please message on [Discord](https://discord.com/invite/u9yhsFmEkB) |
| 4xxx/Radeon VII  | ❓ Untested (WIP)  | N/A  | gfx90x. Implemented but need testers, please message on [Discord](https://discord.com/invite/u9yhsFmEkB) |
| 5xx/Polaris  | ❓ Untested (WIP)  | N/A  | gfx80x. Implemented but need testers, please message on [Discord](https://discord.com/invite/u9yhsFmEkB) |

**Apple:**

| Series  | Supported? |Notes  |
|---|---|---|
| M1/M2/M3/M4  | ✅ Yes  | 'mps' backend  |
| AMD on Intel Mac  | ❓ Untested (WIP)  | Implemented but need testers, please message on [Discord](https://discord.com/invite/u9yhsFmEkB) |

**Intel:**

| Series  | Supported? | OS | Notes  |
|---|---|---|---|
| Arc  | ❓ Untested (WIP)  | Win/Linux  | Implemented but need testers, please message on [Discord](https://discord.com/invite/u9yhsFmEkB). Backends: 'xpu' or DirectML or [ipex](https://github.com/intel/intel-extension-for-pytorch) |


# FAQ
## Why can't I just run 'pip install torch'?
`pip install torch` installs the CPU-only version of torch, so it won't utilize your GPU's capabilities.

## Why can't I just install torch-for-ROCm directly to support AMD?
Different models of AMD cards require different LLVM targets, and sometimes different ROCm versions. And ROCm currently doesn't work on Windows, so AMD on Windows is best served (currently) with DirectML.

And plenty of AMD cards work with ROCm (even when they aren't in the official list of supported cards). Information about these cards (for e.g. the LLVM target to use) is pretty scattered.

`torchruntime` deals with this complexity for your convenience.

# Contributing
📢 I'm looking for contributions in these specific areas:
- More testing on consumer AMD GPUs.
- More support for older AMD GPUs. Explore: Compile and host PyTorch wheels and rocm (on GitHub) for older AMD gpus (e.g. 580/590/Polaris) with the required patches.
- Intel GPUs.
- Testing on professional AMD GPUs (e.g. the Instinct series).
- An easy-to-run benchmark script (that people can run to check the level of compatibility on their platform).

Please message on the [Discord community](https://discord.com/invite/u9yhsFmEkB) if you have AMD or Intel GPUs, and would like to help with testing or adding support for them! Thanks!

# Credits
* Code contributors on [Easy Diffusion](https://github.com/easydiffusion/easydiffusion).
* Users on [Easy Diffusion's Discord](https://discord.com/invite/u9yhsFmEkB) who've helped with testing on various GPUs.
* [PCI Database](https://github.com/pciutils/pciids/) automatically generated from the PCI ID Database at http://pci-ids.ucw.cz

# More resources
* [AMD GPU LLVM Architectures](https://web.archive.org/web/20241228163540/https://llvm.org/docs/AMDGPUUsage.html#processors)
* [Status of ROCm support for AMD Navi 1](https://github.com/ROCm/ROCm/issues/2527)
* [Torch support for ROCm 6.2 on AMD Navi 1](https://github.com/pytorch/pytorch/issues/132570#issuecomment-2313071756)
* [ROCmLibs-for-gfx1103-AMD780M-APU](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU)
