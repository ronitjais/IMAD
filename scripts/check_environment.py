"""Report local Python, package, CUDA, and GPU availability without side effects."""
import importlib.util
import json
import platform
import sys

result = {"python": sys.version, "platform": platform.platform(), "packages": {}}
for name in ("torch", "torchvision", "PIL", "numpy", "sklearn", "yaml", "pytest"):
    result["packages"][name] = importlib.util.find_spec(name) is not None
try:
    import torch
    result["pytorch"] = torch.__version__
    result["cuda_available"] = torch.cuda.is_available()
    result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except ImportError:
    result["cuda_available"] = False
    result["gpu"] = None
print(json.dumps(result, indent=2))
