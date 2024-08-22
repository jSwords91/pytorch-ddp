## **PyTorch Distributed Training**

### Create Virtual Environment

```python -m venv .venv```

```.venv\Scripts\activate```

Press Ctrl+Shift+P, type "Python: Select Interpreter", and choose the interpreter in the .venv folder.


### Runpod

Sign up with [this link](https://runpod.io?ref=diuwf4xe)

Add funds to your account.

Then create an new API key. Copy it and save it somewhere. E.g. a ```.env``` file (make sure to add a ```.gitignore```)

## **Troubleshooting**

### Check CUDA version
```nvcc --version```

### Check local GPU spec.
```nvidia-smi```

### Check Python version

I had to use 3.9

```python --version```

### Installing PyTorch

https://pytorch.org/get-started/locally/

### CUDA Toolkit

Check version matches your local GPU?

https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_network


