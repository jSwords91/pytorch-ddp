# PyTorch Distributed Data Parallel (DDP)

This repository provides a step-by-step guide and examples for implementing distributed training in PyTorch, ranging from single GPU setups to multi-node clusters. The goal is to demonstrate the progression from simple to complex distributed training scenarios.

## Contents

1. **Single GPU Training**
   - Basic PyTorch training on a single GPU
   - Demonstrates fundamental concepts and serves as a baseline

2. **Multi-GPU Training on a Single Machine**
   - Utilizes PyTorch's DistributedDataParallel (DDP)
   - Shows how to parallelize training across multiple GPUs on one machine

3. **Multi-Node Training on a Slurm Cluster**
   - Advanced setup for distributed training across multiple machines
   - Demonstrates integration with Slurm workload manager

## Key Features

- Detailed setup instructions for each scenario
- Example scripts with clear explanations
- Practical tips for efficient distributed training
- Demonstration on both synthetic data and a realistic recommendation system use case

## Getting Started

Start with the single GPU example and progress through the more complex setups as you become comfortable with the concepts.

## Requirements

- PyTorch (with CUDA support for GPU training)
- torchrun (for launching distributed jobs)
- Access to GPU resources (local or cloud-based)
- Slurm cluster access (for multi-node examples)



~ still in progress ~


I'll demo each scenario on fake data, and on a more realistic use case (Recommendation System).

OTHER TO DO:
* Add Opt to save/load funcs
* Demo sharding of data for cases where we can't fit the dataset in memory


## Examples

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

```torchrun --standalone --nproc_per_node=4 multi_gpu_torchrun.py 50 10```

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