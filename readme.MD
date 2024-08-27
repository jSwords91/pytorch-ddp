## **PyTorch Distributed Training**

~ still in progress ~

Based on the excellent PyTorch tutorial, we'll go through PyTorch DistributedDataParallel (DDP) -- which enables data parallel training in PyTorch. Data parallelism is a way to process multiple data batches across multiple devices simultaneously to achieve better performance.

As good as that tutorial was, like most others, it hid the infrastructure set-up which is often a real pain-point for those getting started.

I'll go over **three** methods, including the setup steps:

    1. Single GPU 

    2. Multi GPU

    3. Multi-node on a Slurm Cluster [TO DO]

I'll demo each scenario on fake data, and on a more realistic use case (Recommendation System).

OTHER TO DO:
* Add Opt to save/load funcs


1. Single GPU

This is non-distributed and is how most will get started.

a) You can use your local machine GPU, run ```nvidia-smi``` in the terminal to view details.

simply run ```python single_gpu.py```

If you face issues see troubleshooting below. First check is to uninstall torch and ensure you install the CUDA enabled version.

b) Run on a remote GPU, e.g. on [RunPod](https://runpod.io?ref=diuwf4xe)

I'll come back to that in section 2...

2. Multi GPU

Most will have to use GPU-as-a-service platforms. I use  [RunPod](https://runpod.io?ref=diuwf4xe).

High level steps:

* Sign up and add some funds to your account, e.g. a few pounds/dollars for now.

* Ensure you have the Remote SHH extension installed in VSCode.

* Next we need an SSH key. If you don't have one, these are the steps:

    1. Create a key pair in a terminal or powershell window as follows:
    ```ssh-keygen -t ed25519```

    2. Get your public key (you can use the following command if you used the defaults)

    Windows: ```type %USERPROFILE%\.ssh\id_ed25519.pub```

    Mac: ```cat ~/.ssh/id_ed25519.pub```


* The next steps and following RunPod setup are shown well in this [video](https://www.youtube.com/watch?v=vEVDoW-uMHI)

You could of course select a pod and provision only one GPU, which is a good option if you're model fits and you just need more horsepower.

However for Multi GPU training, simply provision more GPUs for your chosen pod.

Once your pod is running, follow the instructions in the video above to connect through VSCode.

Once you have your remote session running, you need to ```git clone``` the repo.

Next, open the folder where the repo has been written to. 

Then open the terminal and run:

```torchrun --standalone --nproc_per_node=4 multigpu_torchrun.py 50 10```

The ```--nproc_per_node=4``` refers to four GPUs, if I had two, I'd put ```--nproc_per_node=2```.

The script should run on all specified nodes.

3. Mutli-node on a Slurm Cluster

**--Work-in-Progress--**



# **Setup Helpers**

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

I had to use 3.9 to ensure compatibility with CUDA.

```python --version```

### Installing PyTorch

https://pytorch.org/get-started/locally/


### **Downloading Movielens**

In PowerShell

```Invoke-WebRequest -Uri https://files.grouplens.org/datasets/movielens/ml-latest-small.zip -OutFile ml-latest-small.zip
Expand-Archive -Path ml-latest-small.zip -DestinationPath .
Remove-Item ml-latest-small.zip
Rename-Item -Path ml-latest-small -NewName movielens_small```