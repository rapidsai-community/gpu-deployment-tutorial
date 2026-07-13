# GPU deployment Tutorial - SciPy MN 2026

This repository contains materials for the tutorial:
**Deploying and debugging GPU accelerated Python workloads**

## Running the tutorial

### Brev

For the in-person version of this tutorial we will use [NVIDIA Brev](https://brev.nvidia.com/)

### Content

| Module | Link |
| 0 - Introduction Slides | [introduction-to-gpu-stack.html](https://rapidsai-community.github.io/gpu-deployment-tutorial/introduction/introduction-to-gpu-stack.html) |
| 1 - Setup Brev | [1-setup.md](https://rapidsai-community.github.io/gpu-deployment-tutorial/1-setup.html) |
| 2 - Setting Up Your GPU Python Environment | [2-environments.md](https://rapidsai-community.github.io/gpu-deployment-tutorial/2-environments.html) |
| 3 - Verifying Your Environment with the RAPIDS CLI | [3-rapids-cli-verify-env.md](https://rapidsai-community.github.io/gpu-deployment-tutorial/3-rapids-cli-verify-env.html) |
| 4 - Easy GPU Acceleration Wins with RAPIDS Accelerators | [4-acceleration-examples.md](https://rapidsai-community.github.io/gpu-deployment-tutorial/4-acceleration-examples.html) |
| 5 - Monitoring and Debugging GPU Python Workloads | [5-monitoring-and-debugging.md](https://rapidsai-community.github.io/gpu-deployment-tutorial/5-monitoring-and-debugging.html) |

### 1-Setup Brev

- Covers how to get access to a cloud GPU using [NVIDIA Brev](https://brev.nvidia.com).
- Walks through launching a VM (choosing GPU type, provider, and deploying).
- Explains how to connect to the VM via SSH using the `brev` CLI on macOS, Linux, and Windows.

### 2-Setting Up Your GPU Python Environment

- Guides users through installing Python packages for GPU workloads.
- Supports three package managers: `uv`/`pip`, `conda`, and `pixi` via config files (`pyproject.toml`, `environment.yaml`, `pixi.toml`) to manage dependencies— pick one and follow that path.
- Covers checking for core CUDA libraries and creating isolated, reproducible environments.

### 3-Verifying Your Environment with the RAPIDS CLI

- Introduces two commands to validate a GPU environment after setup: `rapids doctor` and `rapids debug`.
- `rapids doctor` runs health checks on the driver, CUDA toolkit, GPU compute capability, and NVLink.
- Failed checks return actionable messages explaining how to fix the issue.
- Showcases how libraries can register their own checks, using `cuml` as an example.
- `rapids debug` produces a full snapshot of driver versions, installed packages, and build tools.
- `rapids debug --json` outputs a machine-readable version, useful for bug reports and sharing environment state.

### 4-Easy GPU Acceleration Wins with RAPIDS Accelerators

- Shows how to get GPU speedups on existing Python code with zero rewrites.
- Shows how to use the accelerators via command line and in notebooks.
- Discusses built-in `--profile` and `--line-profile` flags show which operations ran on the GPU vs. CPU.
- Introduces JupyterLab NVDashboard GPU Accelerators toggle that enables the same zero-code acceleration interactively in notebooks.

### 5-Monitoring and Debugging GPU Python Workloads

- A hands-on diagnostic guide for when GPU code runs but doesn't deliver the expected speedup.
- Covers a progression of tools, each answering a different question about what the GPU is doing.
- `nvidia-smi` and `nvtop` confirm driver-level visibility and live GPU utilization.
- `cProfile` + `SnakeViz` identify CPU-side bottlenecks before any GPU work begins.
- Nsight Systems (`nsys`) produces a full CPU/GPU timeline showing transfers, kernel launches, and idle gaps.
- Uses a real climate data workload (`xarray` EOF analysis) to walk through two common pitfalls:
  - Moving data back to the CPU before the expensive operation.
  - Naively porting a Python loop to the GPU, causing thousands of tiny kernel launches.
- Shows how to identify and fix both issues using profiler output and CUDA timeline traces.
