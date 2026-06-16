# Verify your environment with the RAPIDS CLI

You just set up a GPU environment but before you build on it, two questions are worth answering:

- Did it actually work?
- What exactly did you get?

[RAPIDS CLI](https://github.com/rapidsai/rapids-cli) allows you to answer these questions with two commands.

- `rapids doctor` tells you whether your environment is healthy,
- `rapids debug` tells you exactly what you have in that environment.

## What the RAPIDS CLI does

The RAPIDS CLI bundles the checks you would otherwise run by hand (the GPU driver, the CUDA toolkit and how it lines up with the driver, your Python and library versions, and where CUDA lives) into those two commands. It is also extensible, RAPIDS libraries register their own checks through a plugin system (the `rapids_doctor_check` entry point), and `rapids doctor` discovers and runs them automatically when those libraries are installed.

A few of the pieces it inspects, explained:

- **NVIDIA driver:** The kernel driver that lets software talk to the GPU; its version is the number `nvidia-smi` reports.
- **CUDA driver API version:** The newest CUDA release that driver can support (drivers are backward compatible).
- **CUDA toolkit / runtime:** The CUDA libraries your packages were built against and load at run time, which cannot be newer than the driver supports.
- **[Compute capability](https://developer.nvidia.com/cuda-gpus):** A GPU's hardware feature level; RAPIDS requires 7.0 or higher.
- **[NVLink](https://www.nvidia.com/en-us/data-center/nvlink/):** The high-speed interconnect between GPUs on multi-GPU machines.

Install the RAPIDS CLI into the same environment as your RAPIDS packages (the one you set up earlier), so it reports on the environment you are actually using. With uv:

```bash
uv pip install rapids-cli
```

Or with conda:

```bash
conda install rapidsai::rapids-cli
```

## `rapids doctor`: is my environment healthy?

One command runs a battery of health checks and tells you whether the driver, CUDA toolkit, GPU, memory, and (on multi-GPU boxes) NVLink are consistent with each other.

```console
$ rapids doctor
🧑‍⚕️ Performing REQUIRED health check for RAPIDS
Running checks
All checks passed!
```

Each core check answers one question:

- **GPU present.** Is there at least one GPU, and does it meet the compute capability RAPIDS requires (7.0 or higher)?
- **CUDA driver.** Can we read the CUDA version the driver supports?
- **CUDA toolkit vs driver.** Is the toolkit no newer than the driver supports?
- **Memory ratio.** Is there at least roughly twice as much system memory as total GPU memory? Advisory (handy for Dask), not a hard requirement.
- **NVLink.** On machines with two or more GPUs, are the NVLink connections active?

A failed check doesn't just say "failed." It prints an actionable message telling you how to fix it. For a toolkit that is newer than the driver supports, for example, it tells you to either update the driver or recreate the environment with matching CUDA packages, and links to the CUDA compatibility docs.

Some checks raise a non-fatal warning instead of failing. The memory-ratio check is a good example: it flags a less-than-ideal setup but still lets the overall run pass.

```console
$ rapids doctor
🧑‍⚕️ Performing REQUIRED health check for RAPIDS
Running checks
Warning: System Memory to total GPU Memory ratio not at least 2:1 ratio. It is recommended to have double the system memory to GPU memory for optimal performance.
All checks passed!
```

Add `--verbose` to see what `doctor` discovered and the value behind each check:

```console
$ rapids doctor --verbose
🧑‍⚕️ Performing REQUIRED health check for RAPIDS
Discovering checks
Found check 'cuda' provided by 'rapids_cli.doctor.checks.cuda_driver:cuda_check'
Found check 'cuda_toolkit' provided by 'rapids_cli.doctor.checks.cuda_toolkit:cuda_toolkit_check'
Found check 'gpu' provided by 'rapids_cli.doctor.checks.gpu:gpu_check'
Found check 'gpu_compute_capability' provided by 'rapids_cli.doctor.checks.gpu:check_gpu_compute_capability'
Found check 'memory_to_gpu_ratio' provided by 'rapids_cli.doctor.checks.memory:check_memory_to_gpu_ratio'
Found check 'nvlink_status' provided by 'rapids_cli.doctor.checks.nvlink:check_nvlink_status'
Discovered 6 checks
Running checks
Warning: System Memory to total GPU Memory ratio not at least 2:1 ratio. It is recommended to have double the system memory to GPU memory for optimal performance.
cuda_toolkit_check: CUDA toolkit OK (CUDA 13). Driver supports CUDA 13.
gpu_check: GPU(s) detected: 1
All checks passed!
```

Discovery is the key part: every check, including the six built-ins above, is registered through the `rapids_doctor_check` entry point, and `doctor` runs whatever it finds, including checks shipped by RAPIDS libraries.

## Library checks

RAPIDS libraries register their own checks through the same `rapids_doctor_check` entry point, so installing one adds scoped, library-level smoketests to `rapids doctor` automatically. Each runs a tiny real operation to confirm the library actually works end to end on this machine, not just that the environment looks compatible.

With cuML installed, for example, its four checks are discovered alongside the built-ins:

```console
$ rapids doctor --verbose
🧑‍⚕️ Performing REQUIRED health check for RAPIDS
Discovering checks
...
Found check 'cuml_import' provided by 'cuml.health_checks:import_check'
Found check 'cuml_functional' provided by 'cuml.health_checks:functional_check'
Found check 'cuml_accel_basic' provided by 'cuml.health_checks:accel_basic_check'
Found check 'cuml_accel_cli' provided by 'cuml.health_checks:accel_cli_check'
Discovered 10 checks
...
```

Each verifies something concrete:

- **`cuml_import`.** cuML imports cleanly.
- **`cuml_functional`.** A `LinearRegression` can fit and predict.
- **`cuml_accel_basic`.** `cuml.accel` installs and intercepts scikit-learn.
- **`cuml_accel_cli`.** `python -m cuml.accel` runs scikit-learn code on the GPU.

cuGraph ships a `cugraph_smoke_check` the same way. The more such libraries you install, the more `rapids doctor` answers not just "is my environment compatible?" but "does every RAPIDS library actually run here?"

## `rapids debug`: what exactly do I have?

Where `doctor` gives a verdict, `debug` gives the full picture of the machine: a report you can read or share.

```console
$ rapids debug
RAPIDS Debug Information
...
Driver Version
580.159.03

Cuda Version
13.0

Cuda Runtime Path
/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/include

System Ctk
[]
...
Package Versions
┌───────────────┬──────────┐
│ cuda-bindings │ 13.3.1   │
│ cuda-toolkit  │ 13.3.0   │
│ cupy-cuda13x  │ 14.1.1   │
│ numpy         │ 2.2.6    │
│ pandas        │ 2.3.3    │
│ rapids-cli    │ 0.2.0    │
│ xarray        │ 2025.6.1 │
│ ...           │ ...      │
└───────────────┴──────────┘
...
Tools
┌───────┬──────────────────────────────────────────────────────────────────┐
│ pip   │ pip 22.0.2 from /usr/lib/python3/dist-packages/pip (python 3.10) │
│ uv    │ uv 0.11.21 (x86_64-unknown-linux-gnu)                            │
│ g++   │ g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0                      │
│ nvcc  │ None                                                             │
│ cmake │ None                                                             │
└───────┴──────────────────────────────────────────────────────────────────┘
...
```

It prints a lot, including the full `nvidia-smi` table, your OS details, and every installed package. The fields that matter most after an install:

- **Driver and CUDA versions.** `Driver Version` is the NVIDIA kernel driver (the number `nvidia-smi` shows); `Cuda Version` (13.0 here) is the newest CUDA that driver supports. A toolkit newer than this is the classic broken install, and it is exactly what the toolkit check in `rapids doctor` catches.
- **Toolkit path.** `Cuda Runtime Path` is where your environment found the CUDA runtime, and `System Ctk` lists any toolkits installed under `/usr/local/cuda*`. On this VM `System Ctk` is empty and the runtime path points inside `site-packages`, which tells you CUDA came from pip wheels (the `cuda-toolkit` and `nvidia-cuda-*` packages) rather than a system install. Together they tell you which CUDA you are actually using when more than one could be present.
- **Library and tool versions.** `Package Versions` lists every installed Python package and version, and `Tools` reports the build and packaging tools on your `PATH`. A tool that isn't installed shows as `None`, like `nvcc` and `cmake` here. This is the quickest way to confirm your RAPIDS libraries and build tools are the versions you expect.

Trimmed to the same fields, `rapids debug --json` gives you a machine-readable version:

```console
$ rapids debug --json
{
    "driver_version": "580.159.03",
    "cuda_version": "13.0",
    "cuda_runtime_path": "/home/ubuntu/.local/lib/python3.10/site-packages/nvidia/cu13/include",
    "system_ctk": [],
    "package_versions": {
        "cuda-bindings": "13.3.1",
        "cuda-toolkit": "13.3.0",
        "cupy-cuda13x": "14.1.1",
        "rapids-cli": "0.2.0"
    },
    "tools": {
        "pip": "pip 22.0.2 from /usr/lib/python3/dist-packages/pip (python 3.10)",
        "uv": "uv 0.11.21 (x86_64-unknown-linux-gnu)",
        "nvcc": null,
        "cmake": null
    }
}
```

This is what you paste into a bug report or GitHub issue when you ask for help. It's the same dump the team uses to track environments across platforms.

## When to reach for which

Run `rapids doctor` when you want a yes or no on whether your environment is healthy. Run `rapids debug` (or `rapids debug --json`) when you need the details, or when you want to share exactly what you have while asking for help.
