# We have a GPU, now what?

We have a GPU, but now we need to install libraries to be able to use it. How do
we do this?

In this tutorial we will be working with python libraries to showcase some
common workflows that you might have encounter as part of your Data Science,
Machine Learning, or Scientific Computing journey.

Keep in mind that the goal of this tutorial is to teach you how to monitor, debug
and find performance, so the examples are set to highlight these aspects.

## How to install packages

When it comes into performing some work on the GPU, we highly recommend that you
install packages via in environments. This makes workflows more reproducible, and
easier to track installations problems when they occur.  

If you use python libraries, you probably use one or more of these in your local
setups:

- `conda`
- `uv`/`pip`
- `pixi`

This tutorial will show a path on how to use each of the package managers. That
being said, choose one and follow teh instructions just for that path, do not mix
and match since this will create conflicts.

**NOTE:** In the live version, we will ask for a a show of hands to choose one.  

### core CUDA libraries check

We need the core CUDA libraries in order to run any CUDA code. Often these will be
installed at the system level in `/usr/local/cuda`. Let's check that:

```bash
ls -ld /usr/local/cuda*
```

If these are missing we need to decide how to get those dependencies. The way we
do this is different depending on whether we want to use `pip/uv` or `conda` or `pixi`
for our Python package manager.

### uv/pip

> [!NOTE]
> (uv/pip CUDA caveats):
> The pip/uv ecosystem is moving toward fully self-contained CUDA wheels, similar to conda,
> but this transition is still in progress.
>
> Some RAPIDS libraries (e.g., cudf, cuml) provide CUDA-enabled wheels (*-cuXX) that
> bundle the required runtime and work without a system CUDA toolkit. However, other
> dependencies and older versions (notably cupy < 14) may still require CUDA to be
> installed on the system.
>
> In practice, using recent packages (where dependencies on cupy >= 14) should allow
> a fully pip-based setup without system CUDA, but this is still being validated
> across all libraries.
>
> We recommend start with CUDA-enabled wheels and no system CUDA. If you encounter
> missing CUDA errors, check package versions first, then fall back to installing
> a system CUDA toolkit if needed.

#### Installing uv

`uv` by Astral, has become a popular choice to install packages via pip. In this
tutorial we will show case how you can create your environment using `uv`.

> [!NOTE]
> Brev already has `uv` installed and manages an active `.venv` at `~/.venv` that powers the JupyterLab session. We want to install our packages into that same environment so they are available in our notebooks without any extra kernel configuration.

**Not on Brev?**

First install `uv` following the [Astral documentation](https://docs.astral.sh/uv/getting-started/installation/#installation-methods):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You'll need to source your `.bashrc` to make `uv` available in your current shell:

```bash
source ~/.bashrc
```

#### Creating an Environment

When we check the driver version (`nvidia-smi`) we noticed that we have CUDA 13
drivers, so we will install the `cu13` versions of these packages.

We'll manage our environment using a `pyproject.toml` that lives under `envs/` in this repository. Take a look at `envs/pyproject.toml` to see the full list of dependencies that we will be using along this tutorial.

Navigate to that directory and sync all dependencies into the Brev `.venv`:

```bash
cd envs/
UV_PROJECT_ENVIRONMENT=~/.venv uv sync
```

> [!NOTE]
> By default, `uv sync` creates and manages its own `.venv` inside the project directory, regardless of which environment is activated in your shell. Setting `UV_PROJECT_ENVIRONMENT=~/.venv` tells `uv` to target the existing Brev virtual environment instead.
>
> If you are **not** on Brev and don't have a pre-existing environment, let `uv` create one for you inside the project directory:
>
> ```bash
> cd envs/
> uv sync
> source .venv/bin/activate
> ```
>
> `uv sync` will create `.venv/` automatically if it doesn't exist, then install all dependencies into it.

#### Testing the Installation

Launch the Python interpreter and test with some cuDF code:

```python
import cudf

# Create a cuDF DataFrame
data = {'col1': [1, 2, 3, 4], 'col2': [10, 20, 30, 40]}
df = cudf.DataFrame(data)

# Perform an operation on a DataFrame column
df['col3'] = df['col1'] * df['col2']
df
```

We will be using this environment in the following sections.

#### Important Limitation

When installing nightly or pre-release versions of packages, `uv` has an all-or-nothing strategy. It requires more explicit configuration when working with nightlies or pre-releases, and failing to do so can generate version conflicts and installation errors that are less common with `pip`. For more information, see the [uv pre-release compatibility documentation](https://docs.astral.sh/uv/pip/compatibility/#pre-release-compatibility).

### conda

#### Conda

When installing libraries with conda each individual CUDA library can be installed as a conda package. So we don't need to ensure any of the CUDA libraries already exist in `/usr/local/cuda`.

If you prefer to use `conda` then we need to install it first.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

bash Miniforge3-$(uname)-$(uname -m).sh  # Follow the prompts and choose yes to update your shell profile to automatically initialize conda
```

> [!NOTE]
> You'll need to source your `.bashrc` to make `conda` available in your current shell:

```bash
source ~/.bashrc
```

We'll create the environment from the `environment.yaml` file in `envs/`. Take a look at [envs/environment.yaml](envs/environment.yaml) to see the full list of dependencies.

```bash
conda env create -f envs/environment.yaml
conda activate tutorial-env
python -m ipykernel install --user --name tutorial-env --display-name "Scipy GPU deployment tutorial env"
```

Then we can import `cudf` and allocate some GPU memory

```python
import cudf
s = cudf.Series(['a', 'aa', 'b'])
s.apply(lambda x: len(x))
```

### Installing the JupyterLab NV Dashboard Extension

We will be using the [jupyterlab-nvdashboard](https://github.com/rapidsai/jupyterlab-nvdashboard) extension to view GPU metrics directly in the JupyterLab interface.

This extension must be installed separately from the conda environment. JupyterLab extensions need to be installed where the **JupyterLab server** runs, not where individual kernels run — so installing it inside `tutorial-env` would make it available as a Python package, but the JupyterLab UI would never see it.

On Brev, the JupyterLab server runs from `/home/ubuntu/.venv/`, the system `uv` environment. From a terminal inside JupyterLab, first deactivate your conda environment so the system `uv` is used, then install and restart:

```bash
conda deactivate
echo $VIRTUAL_ENV  # should show /home/ubuntu/.venv
uv pip install jupyterlab-nvdashboard
sudo systemctl restart jupyter.service
```

Exit and reopen the notebook, or refresh your browser. The GPU dashboard panels will now be available in the JupyterLab sidebar.
