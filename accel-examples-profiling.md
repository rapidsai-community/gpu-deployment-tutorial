# Easy GPU Acceleration Wins with RAPIDS

This section shows how to take existing CPU Python workflows and try GPU
acceleration without rewriting the application. We will stay in the terminal and
run scripts directly. Notebook usage is covered separately.

You will try two accelerators:

- `cudf.pandas` for pandas-style dataframe work.
- `cuml.accel` for scikit-learn-style machine learning work.

The goal is not to tune every line of code. The goal is to quickly answer:

- Does my existing script run with the accelerator?
- Which parts ran on the GPU?
- Did the runtime improve enough to investigate further?

## Setup checkup

```bash
nvidia-smi
```

Make sure you have an active environment that has `cudf`, `cuml`, `pandas` and
`sklearn`.

Confirm that these packages are available in the Python environment you are
using:

```bash
python -c "import cudf, cuml, pandas, sklearn; print('environment is ready')"
```

Download the dataset used by the pandas workflow:

```bash
python scripts/data-setup.py --nyc-parking
```

## pandas Workflow: Baseline CPU Run

Imagine you already have pandas code in a script and want to know whether it can
benefit from a GPU.

First inspect the existing pandas workflow. For example, in `pandas-workflow.py`
we read the NYC parking violations data and runs common dataframe
operations such as `value_counts`, `groupby`, `agg`, datetime extraction,
sorting, and `count`.

Run it normally, to see the cpu baseline:

```bash
python scripts/pandas-workflow.py
```

## Accelerate pandas with `cudf.pandas`

If `cudf` is installed, try the same script with zero code changes:

```bash
python -m cudf.pandas scripts/pandas-workflow.py
```

That is the main workflow: replace `python` with `python -m cudf.pandas`.

Your script still imports `pandas`, but `cudf.pandas` intercepts pandas imports
and uses cuDF on the GPU where possible. Operations that are not supported on
the GPU fall back to pandas on the CPU.

## Understanding cudf.pandas Performance

**Exercise:** What do you notice? When comparing the runs, keep these points in
mind:

- The first GPU run may include startup and GPU context initialization overhead.
- The script reads the parquet file before the timed operations, so the printed
  timings focus on dataframe operations rather than download or file I/O.
- Larger dataframe operations usually show the clearest speedups.
- Very small operations can be dominated by overhead and may not be faster.
- Unsupported pandas operations can fall back to CPU, which is correct but may
  reduce the speedup.

We can, have more insight on what's happening using the the built-in
`cudf.pandas` profilers.

We have the line profiler that shows the source code and how much time each line
spent executing on the GPU and CPU.

TODO: add output pending <https://github.com/rapidsai/cudf/issues/23010>

```bash
python -m cudf.pandas --line-profile scripts/pandas-workflow.py
```

and if we use `--profile`, it generates a report showing which operations used
the GPU and which used the CPU.

```bash
python -m cudf.pandas --profile scripts/pandas-workflow.py
```

```txt  
                                             Total time elapsed: 27.052 seconds  
                                           28 GPU function calls in 2.050 seconds  
                                           1 CPU function calls in 22.172 seconds  

                                                           Stats  

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Function                       ┃ GPU ncalls ┃ GPU cumtime ┃ GPU percall ┃ CPU ncalls ┃ CPU cumtime ┃ CPU percall ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ read_parquet                   │ 1          │ 1.167       │ 1.167       │ 0          │ 0.000       │ 0.000       │
│ DataFrame.__getitem__          │ 3          │ 0.004       │ 0.001       │ 0          │ 0.000       │ 0.000       │
│ DataFrame.value_counts         │ 1          │ 0.237       │ 0.237       │ 0          │ 0.000       │ 0.000       │
│ Series.groupby                 │ 1          │ 0.001       │ 0.001       │ 0          │ 0.000       │ 0.000       │
│ GroupBy.head                   │ 1          │ 0.144       │ 0.144       │ 0          │ 0.000       │ 0.000       │
│ Series.sort_index              │ 1          │ 0.006       │ 0.006       │ 0          │ 0.000       │ 0.000       │
│ Series.reset_index             │ 1          │ 0.001       │ 0.001       │ 0          │ 0.000       │ 0.000       │
│ DataFrame.groupby              │ 3          │ 0.078       │ 0.026       │ 0          │ 0.000       │ 0.000       │
│ DataFrameGroupBy.aggregate     │ 1          │ 0.024       │ 0.024       │ 0          │ 0.000       │ 0.000       │
│ DataFrame.rename               │ 1          │ 0.001       │ 0.001       │ 0          │ 0.000       │ 0.000       │
│ DataFrame.sort_values          │ 1          │ 0.006       │ 0.006       │ 0          │ 0.000       │ 0.000       │
│ NDFrame.astype                 │ 1          │ 0.149       │ 0.149       │ 0          │ 0.000       │ 0.000       │
│ DataFrame.__setitem__          │ 2          │ 0.005       │ 0.002       │ 0          │ 0.000       │ 0.000       │
│ Series                         │ 1          │ 0.000       │ 0.000       │ 0          │ 0.000       │ 0.000       │
│ CombinedDatetimelikeProperties │ 1          │ 0.000       │ 0.000       │ 0          │ 0.000       │ 0.000       │
│ Series.map                     │ 1          │ 0.158       │ 0.158       │ 0          │ 0.000       │ 0.000       │
│ DataFrameGroupBy.__getitem__   │ 1          │ 0.003       │ 0.003       │ 0          │ 0.000       │ 0.000       │
│ GroupBy.count                  │ 1          │ 0.025       │ 0.025       │ 0          │ 0.000       │ 0.000       │
│ Series.sort_values             │ 2          │ 0.011       │ 0.006       │ 0          │ 0.000       │ 0.000       │
│ GroupBy.size                   │ 1          │ 0.021       │ 0.021       │ 0          │ 0.000       │ 0.000       │
│ NDFrame.head                   │ 1          │ 0.001       │ 0.001       │ 0          │ 0.000       │ 0.000       │
│ DataFrame.count                │ 1          │ 0.006       │ 0.006       │ 1          │ 22.172      │ 22.172      │
└────────────────────────────────┴────────────┴─────────────┴─────────────┴────────────┴─────────────┴─────────────┘
Not all pandas operations ran on the GPU. The following functions required CPU fallback:

- DataFrame.count
```

Look for:

- Operations that ran on the GPU.
- Operations that fell back to CPU.
- Lines or functions that dominate total runtime.

If a row-wise or Python-object-heavy operation falls back to CPU, that is often
the next place to simplify the workflow.

## scikit-learn Workflow: Baseline CPU Run

Similarly, we have `cuml.accel` for `scikit-learn`, `UMAP`, and `HDBSCAN`

The `sklearn-workflow.py` script generates a synthetic classification dataset,
trains a `RandomForestClassifier`, predicts on a test split, and prints accuracy.

Run the CPU baseline:

```bash
python scripts/sklearn-workflow.py
```

## Accelerate scikit-learn with `cuml.accel`

If `cuml` is installed, try the same script with zero code changes:

```bash
python -m cuml.accel scripts/sklearn-workflow.py
```

Again, the script still imports from `sklearn`. The accelerator dispatches
supported estimators and methods to cuML on the GPU and falls back to CPU where
needed.

Run both paths a few times:

```bash
for i in 1 2 3; do python scripts/sklearn-workflow.py; done
for i in 1 2 3; do python -m cuml.accel scripts/sklearn-workflow.py; done
```

The accuracy does not need to be bit-for-bit identical between CPU and GPU
implementations. Compare model quality, not internal fitted attributes.

## Understanding scikit-learn Performance

Machine learning acceleration depends on the estimator, hyperparameters, data
size, and data types.

`cuml.accel` also counts with profilers to get better understanding on what's being accelerated:

Use the function profiler for a compact report:

```bash
python -m cuml.accel --profile scripts/sklearn-workflow.py
```

```txt
cuml.accel profile  
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Function                       ┃ GPU calls ┃ GPU time ┃ CPU calls ┃ CPU time ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│ RandomForestClassifier.fit     │         1 │  280.7ms │         0 │       0s │
│ RandomForestClassifier.predict │         1 │     70ms │         0 │       0s │
├────────────────────────────────┼───────────┼──────────┼───────────┼──────────┤
│ Total                          │         2 │  350.8ms │         0 │       0s │
└────────────────────────────────┴───────────┴──────────┴───────────┴──────────┘
```

Use the line profiler when you want per-line detail:

```bash
python -m cuml.accel --line-profile scripts/sklearn-workflow.py
```

```txt
cuml.accel line profile  
┏━━━━┳━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  # ┃ N ┃    Time ┃ GPU % ┃ Source                                                         ┃
┡━━━━╇━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│  1 │ 1 │ 271.6ms │     - │ from sklearn.ensemble import RandomForestClassifier            │
│  2 │ 1 │  73.3ms │     - │ from sklearn.datasets import make_classification               │
│  3 │ 1 │       - │     - │ from sklearn.model_selection import train_test_split           │
│  4 │ 1 │       - │     - │ from sklearn.metrics import accuracy_score                     │
│  5 │ 1 │       - │     - │ from joblib import dump, load                                  │
│  6 │   │         │       │                                                                │
│  7 │ 1 │       - │     - │ import time                                                    │
│  8 │   │         │       │                                                                │
│  9 │   │         │       │ # synthetic dataset dimensions                                 │
│ 10 │ 1 │       - │     - │ n_samples = 100_000                                            │
│ 11 │ 1 │       - │     - │ n_features = 10                                                │
│ 12 │ 1 │       - │     - │ n_classes = 2                                                  │
│ 13 │   │         │       │                                                                │
│ 14 │   │         │       │ # random forest depth and size                                 │
│ 15 │ 1 │       - │     - │ n_estimators = 25                                              │
│ 16 │ 1 │       - │     - │ max_depth = 10                                                 │
│ 17 │   │         │       │                                                                │
│ 18 │ 1 │       - │     - │ start = time.time()                                            │
│ 19 │   │         │       │                                                                │
│ 20 │   │         │       │ # generate synthetic data [ binary classification task ]       │
│ 21 │ 2 │  54.9ms │     - │ X, y = make_classification(                                    │
│ 22 │ 1 │       - │     - │     n_classes=n_classes,                                       │
│ 23 │ 1 │       - │     - │     n_features=n_features,                                     │
│ 24 │ 1 │       - │     - │     n_samples=n_samples,                                       │
│ 25 │   │         │       │ )                                                              │
│ 26 │   │         │       │                                                                │
│ 27 │ 1 │     9ms │     - │ X_train, X_test, y_train, y_test = train_test_split(X, y)      │
│ 28 │   │         │       │                                                                │
│ 29 │ 2 │       - │     - │ model = RandomForestClassifier(                                │
│ 30 │ 1 │       - │     - │     max_depth=max_depth,                                       │
│ 31 │ 1 │       - │     - │     n_estimators=n_estimators,                                 │
│ 32 │   │         │       │ )                                                              │
│ 33 │   │         │       │                                                                │
│ 34 │ 1 │ 272.8ms │  99.0 │ trained_RF = model.fit(X_train, y_train)                       │
│ 35 │   │         │       │                                                                │
│ 36 │ 1 │  70.6ms │  99.0 │ predictions = model.predict(X_test)                            │
│ 37 │   │         │       │                                                                │
│ 38 │ 1 │   4.5ms │     - │ score = accuracy_score(y_test, predictions)                    │
│ 39 │   │         │       │                                                                │
│ 40 │ 1 │       - │     - │ end = time.time()                                              │
│ 41 │   │         │       │                                                                │
│ 42 │ 1 │       - │     - │ print("sklearn accuracy:", score)                              │
│ 43 │ 1 │       - │     - │ print(f"Total elapsed time for RF: {end - start:.4f} seconds") │
└────┴───┴─────────┴───────┴────────────────────────────────────────────────────────────────┘
Ran in 758ms, 45.1% on GPU  
```

Do not use profiler runs as benchmark numbers. Profilers add overhead. Use them
to understand dispatch behavior, then benchmark with the plain accelerated
command.

## Watch the GPU with Jupyterlab NVDashboard

If you work on Jupyter notebooks, the JupyterLab NVDashboard extension is a great
tool to watch some GPU metrics like memory and utilization.

As of `jupyterlab-nvdashboard >= 0.14`, the extension includes a **GPU Accelerators**
panel with a GPU accelerator activator button that lets you enable GPU-backed execution
with zero code changes. When active, your existing pandas code runs on the GPU (via
cudf-pandas), and/or your scikit-learn, UMAP, and HDBSCAN code runs on the GPU (via
cuml-accel). Accelerators are shown only when the corresponding dependencies are
installed.

Install the dashboard into the Python environment used by the Brev Jupyter
server.

Note: if you are following this tutorial in order, this part should have been
taken care of as part of the environments setup.

```bash
python -m pip install jupyterlab_nvdashboard
sudo systemctl restart jupyter
```

### Example

Open the notebook in the [notebooks/](notebooks/) directory to follow along
interactively.

Before running any code, open the nvDashboard panels for **GPU Utilization**
and **GPU Memory**. You can find them in the JupyterLab sidebar under the
chart icon added by the extension.

Run the notebook cells as-is and watch the dashboards. You should notice
**no GPU activity** — the CPU handles all the work and the GPU metrics stay
flat.

Now enable the `cudf.pandas` toggle (the cell at the top of the notebook that
activates the accelerator) and run the notebook again. Watch the dashboards:
GPU utilization and memory will spike as pandas operations are transparently
offloaded to the GPU.

## Takeaways

- `cudf.pandas` is the fastest way to try GPU acceleration on existing pandas code.
- `cuml.accel` is the fastest way to try GPU acceleration on supported scikit-learn,
UMAP, and HDBSCAN workflows.
- Built-in accelerator profilers explain GPU execution and CPU fallback.

## References

- [cuDF pandas accelerator usage](https://docs.rapids.ai/api/cudf/stable/cudf_pandas/usage/)
- [cuML zero code change acceleration](https://docs.rapids.ai/api/cuml/stable/cuml-accel/)
- [cuML accelerator logging and profiling](https://docs.rapids.ai/api/cuml/stable/cuml-accel/logging-and-profiling/)
- [cuML accelerator limitations](https://docs.rapids.ai/api/cuml/stable/cuml-accel/limitations/)
- [jupyterlab-nvdashboard](https://github.com/rapidsai/jupyterlab-nvdashboard)
