# API
Import using:
```py
import torchruntime
```

## Install torch packages
You can use the command line:
`python -m torchruntime install <optional list of package names and versions>`

Or you can use the library:
```py
torchruntime.install(["torch", "torchvision<0.20"])
```

## Get device info
You can use the device database built into `torchruntime` for your projects:
```py
from torchruntime.device_db import get_gpus

discrete_gpus = get_gpus()  # Returns a list of `torchruntime.device_db.GPU` instances containing the fields: vendor_id, vendor_name, device_id, device_name, is_discrete
```

**Important:** This API could break in the future, so if you're writing a program using this, please open a new Issue on this repo and let me know what you're trying to do.

## Get torch platform
This will provide you the recommended torch platform to use for the PC. It analyzes the GPUs and OS on the PC, and suggests the most-performant version of torch for that.

E.g. `cu124` or `rocm6.1` or `directml` or `ipex` or `xpu` or `cpu`.

```py
from torchruntime.platform_detection import get_torch_platform

torch_platform = get_torch_platform()
```
