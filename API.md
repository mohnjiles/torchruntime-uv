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
from torchruntime.device_db import get_discrete_gpus

discrete_gpus = get_discrete_gpus()  # Returns a list of tuples containing (vendor_id, vendor_name, device_id, device_name)
```

**Important:** This API could break in the future, so if you're writing a program using this, please open a new Issue on this repo and let me know what you're trying to do.
