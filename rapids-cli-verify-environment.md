# Verify your environment with the RAPIDS CLI

You just set up a GPU environment. Before you build on it, two questions are worth answering: did it actually work, and what exactly did you get? The manual answer is a scatter of commands: `nvidia-smi`, `python --version`, some hand-written pynvml, and poking around `/usr/local/cuda`. The RAPIDS CLI collapses that into two commands. `rapids doctor` tells you whether your environment is healthy, and `rapids debug` tells you exactly what you have.

## What the RAPIDS CLI does

The RAPIDS CLI verifies that a RAPIDS environment is set up correctly and reports what you have installed. It bundles the checks you would otherwise run by hand (the GPU driver, the CUDA toolkit and how it lines up with the driver, your Python and library versions, and where CUDA lives) into two commands: `rapids doctor` and `rapids debug`.

Install it with pip or uv:

```bash
pip install rapids-cli        # or: uv pip install rapids-cli
```

On Brev, activate the uv virtual environment first (`source .venv/bin/activate`) so the install and the environment it reports on are the same one.

## `rapids doctor`: is my environment healthy?

One command runs a battery of health checks and tells you whether the driver, CUDA toolkit, GPU, memory, and (on multi-GPU boxes) NVLink are consistent with each other.

```console
$ rapids doctor
🧑‍⚕️ Performing REQUIRED health check for RAPIDS
Running checks
All checks passed!
```

Each core check answers one question:

- **GPU present.** Is there at least one GPU, and does it meet the minimum compute capability (7.0) that RAPIDS requires?
- **CUDA driver.** Can we read the CUDA version the GPU driver supports?
- **CUDA toolkit vs driver.** Is the installed CUDA toolkit something the driver actually supports? Drivers are backward compatible, so this fails only when the toolkit is newer than the driver, which is the classic broken install.
- **Memory ratio.** Is there at least roughly twice as much system memory as total GPU memory? This is an advisory (handy for Dask), not a hard requirement.
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

The verbose run shows two things. First, discovery: every check, including the six built-in ones above, is registered through the `rapids_doctor_check` entry point. RAPIDS libraries like cudf, cuml, and cugraph register their own checks the same way, so the discovered count grows with whatever RAPIDS packages you have installed.

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

- **Driver version and CUDA version.** It helps to keep three things straight. The NVIDIA kernel driver is the low-level driver (the `Driver Version` above, and the number `nvidia-smi` shows). The CUDA driver API version is the newest CUDA that driver can support (the `Cuda Version` above, here 13.0). The CUDA toolkit and runtime is what your libraries were built against and load at run time. A toolkit newer than the driver supports is the classic install problem, and it's exactly what the toolkit check in `rapids doctor` catches.
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

## Future Scope

Today `rapids doctor` mostly verifies that the driver, CUDA, and hardware are compatible with each other. The next step is scoped, library-level smoketests. Each RAPIDS library (cudf, cuml, cugraph, and so on) ships a small check, registered through the same `rapids_doctor_check` entry point, that runs a tiny real operation to confirm the library actually works end to end on this machine. Build a cuDF Series, fit a trivial cuML estimator, construct a small cugraph graph. As those land, `rapids doctor` becomes a single command that answers not just "is my environment compatible?" but "does every RAPIDS library I installed actually run here?" The entry-point mechanism already exists in `rapids-cli` currently, and libraries will be adopting it in the near future.