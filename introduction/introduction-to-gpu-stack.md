---
marp: true
theme: nvidia
paginate: true
---

<!-- _class: title -->
<!-- _paginate: false -->

# Understanding GPUs
## From the *GPU* to *RAPIDS* : how the pieces fit together

**Jaya Venkatesh and Naty Clementi · NVIDIA**
SciPy 2026 · GPU Deployment & Debugging Tutorial

---

# Why bring data science to the GPU?

<div class="cards3">
  <div class="card"><h3>Bigger data</h3><p>Datasets keep outgrowing the power of CPUs</p></div>
  <div class="card"><h3>Parallel by nature</h3><p>Filtering, aggregating, and math over millions of rows are independent</p></div>
  <div class="card"><h3>Familiar APIs</h3><p>You can get there without leaving Python</p></div>
</div>

<br>

> But how does a GPU actually **help you accelerate your code?**

---

<!-- _class: section -->

# 1 · The GPU vs the CPU
> Both process data, just with a different philosophy

---

# A CPU and a GPU are built for different jobs

<div class="center">
<img src="images/cuda-figures/gpu-devotes-more-transistors-to-data-processing.png" style="width:66%;" />
<span class="src">Source: NVIDIA CUDA C++ Programming Guide</span>
</div>

<div class="columns">
<div>

**CPU**: handles **many different tasks**, finishing **each one fast** → *low latency*.

</div>
<div>

**GPU**: runs the **same task across lots of data** all at once → *high throughput*.

</div>
</div>

---

# A few specialists vs many simple workers

<div class="coresrow">
<div class="corebox"><div class="cap">CPU</div>
<img class="ig" src="images/infographics/cpu-cores.svg" />
<div class="muted" style="font-size:0.7em; margin-top:8px;">a few powerful cores</div>
</div>
<div class="corebox"><div class="cap">GPU</div>
<img class="ig" src="images/infographics/gpu-cores.svg" />
<div class="muted" style="font-size:0.7em; margin-top:8px;">many simple cores</div>
</div>
</div>

- CPU cores are **few but powerful**, great at complex, branchy logic
- GPU cores are **many but simple**, built to all do the **same operation** on different data, in parallel
- The more your work splits into **independent pieces**, the more **using the GPU pays off**

---

# Two machines, two separate memories

<div class="xfer">
<div class="dev"><div class="name">CPU</div><div class="cap" style="font-size:0.84em; margin-top:12px;">System RAM<br>large, general-purpose</div></div>
<div class="link" style="font-weight:700; color:var(--nv-grey);">must copy<br>data across →</div>
<div class="dev gpu"><div class="name">GPU</div><div class="cap" style="font-size:0.84em; margin-top:12px;">VRAM<br>its own dedicated memory</div></div>
</div>

<br>

- The GPU is a **separate device** with its **own memory**: it **cannot** read data sitting in system RAM
- Before the GPU can work on your data, that data must be **copied into the GPU's memory**

> So step one of "run it on the GPU" is always: **get the data there.**

---

<!-- _class: bw -->

# The slow part is getting data *across*

<div class="xfer">
<div class="dev"><div class="name">CPU + RAM</div><div class="hw" style="background:var(--nv-grey);"></div><div class="cap">high memory bandwidth</div></div>
<div class="link">PCIe<div class="thin"></div>lower bandwidth</div>
<div class="dev gpu"><div class="name">GPU + VRAM</div><div class="hw"></div><div class="cap">high memory bandwidth</div></div>
</div>

<br>

- **Inside** each device, **memory bandwidth is high**: RAM and VRAM are built to feed their cores fast
- The **interconnect between them (PCIe)** carries data at **far lower bandwidth**
- The expensive part often isn't the *computing*; it's the **host-to-device transfer** across PCIe

---

<!-- _class: section -->

# 2 · How a GPU processes data
> Many small tasks at once, not one big task in order

---

# Parallel work: threads, blocks, grids

The GPU runs **one operation across thousands of elements at once**, one **thread** per item.

<div class="center">
<img src="images/cuda-figures/grid-of-thread-blocks.png" style="width:48%;" />
<span class="src">Source: NVIDIA CUDA C++ Programming Guide</span>
</div>

- Threads are grouped into **blocks**; blocks together form a **grid**
- You describe the work *once*; the GPU runs it across the **whole grid in parallel**
- Perfect for **independent** work, not optimal for **sequential**, step-by-step logic

---

# Why GPU is better for parallel than sequential

<div class="tl">
<div class="tlcol"><div class="h">Sequential</div>
<img class="ig" src="images/infographics/seqbar.svg" />
<div class="t">one after another → time</div>
</div>
<div class="tlcol"><div class="h">Parallel</div>
<img class="ig" src="images/infographics/parstack.svg" />
<div class="t">all at once → done sooner</div>
</div>
</div>

<br>

- When work items are **independent**, their order doesn't matter i.e item B doesn't need item A's result
- A GPU runs a **huge number of independent items at the same time**
- Step-by-step work that must run **in order** can't use that; this isn't where a GPU shines

---

# SMs and warps: where the work runs

<div class="columns">
<div>

- A GPU is made of **Streaming Multiprocessors (SMs)**, its parallel engines
- Your **blocks are distributed across the SMs**; more SMs → more work at once, **automatically**
- Inside an SM, threads run in lock-step groups of **32, called a warp** *(SIMT: Single Instruction, Multiple Threads)*

</div>
<div class="center">
<img src="images/cuda-figures/automatic-scalability.png" style="width:92%;" />
<span class="src">Source: NVIDIA CUDA C++ Programming Guide</span>
</div>
</div>

---

# Compute-bound vs IO-bound

<div class="columns">
<div>

### ✓ Compute-bound: GPU shines
- A matrix multiply, training a model, transforming millions of rows
- **Lots of math** → the cores stay saturated

</div>
<div>

### ✗ IO-bound: GPU waits
- Reading a file off disk, waiting on a network call
- The bottleneck is **waiting**, not math

</div>
</div>

<br>

> Adding compute units only helps when **computation** is the bottleneck. If you're waiting on data, more cores don't make the wait shorter.

---

# So what actually fits a GPU?

<div class="cards">
  <div class="card"><h3>✓ Parallel</h3><p>The same operation over millions of elements</p></div>
  <div class="card"><h3>✗ Sequential</h3><p>Each step depends on the previous one</p></div>
  <div class="card"><h3>✓ Compute-bound</h3><p>Lots of math per byte of data</p></div>
  <div class="card"><h3>✗ IO-bound</h3><p>Time spent waiting on disk / network</p></div>
</div>

> Takeaway: a GPU shines on **big, math-heavy, parallel** work, so the goal is to match your workload to its strengths.

---

<!-- _class: section -->

# 3 · CUDA
> The bridge that lets software use the GPU

---

<!-- _class: cudaslide -->

# What is CUDA, and why C?

<div class="columns">
<div>

**CUDA** is NVIDIA's parallel-computing **platform + programming model**, built on **C/C++** for its low-level speed and control over the hardware.

- You write GPU code as **kernels** in **CUDA C/C++**: standard C++/C plus a few extensions, compiled by `nvcc`
- The **host** (CPU) runs serial code and **launches kernels** on the **device** (GPU)
- Serial host code and parallel kernels **alternate** (see diagram →)

</div>
<div class="center">
<img src="images/cuda-figures/heterogeneous-programming.png" style="width:60%;" />
<span class="src">Source: NVIDIA CUDA C++ Programming Guide</span>
</div>
</div>

---

# A CUDA kernel is just C, with a twist

```cpp
// __global__ marks a function that runs ON the GPU, once per thread
__global__ void add(float *a, float *b, float *c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // which element am I?
    c[i] = a[i] + b[i];                             // one thread, one element
}

// The host launches it across a grid:  <<< blocks, threads-per-block >>>
add<<<blocks, threads>>>(a, b, c);   // every element added in parallel
```

> The `__global__` keyword and the `<<< >>>` launch syntax are the **C extensions** CUDA adds. A sum, a sort, a join all become kernels like this, but **you'll rarely write one**: the layers above ship them ready-made.

---

<!-- _class: section -->

# 4 · CUDA Python & the CUDA Toolkit
> The building blocks above raw CUDA

---

# The CUDA Toolkit: the foundation libraries

Everything above the driver is **built on the Toolkit**. It provides:

<div class="cards">
  <div class="card"><h3>Compiler &amp; runtime</h3><p>nvcc, libcudart, headers</p></div>
  <div class="card"><h3>Math libraries</h3><p>cuBLAS, cuFFT, cuSPARSE, cuSOLVER, cuRAND</p></div>
  <div class="card"><h3>Communication</h3><p>NCCL, NVSHMEM for multi-GPU</p></div>
  <!-- <div class="card"><h3>Why it's central</h3><p>cuDF, cuML &amp; CuPy don't reinvent math, they call these</p></div> -->
</div>

> These hand-tuned libraries are the **ready-made kernels**: the sorts, joins, and math behind your data work come straight from here.

---

# CUDA Python: pick your entrypoint

You don't have to write C++ to use CUDA.

<div class="cards3">
  <div class="card"><h3>CuPy</h3><p>"I have NumPy code" → swap in GPU arrays</p></div>
  <div class="card"><h3>Numba (CUDA)</h3><p>"I need a custom kernel" → write it in Python</p></div>
  <div class="card"><h3>cuda.core / cuda.bindings</h3><p>"I'm building a library / need low-level control"</p></div>
</div>

<br>

> All three reach the **GPU underneath**.

---

<!-- _class: section -->

# 5 · RAPIDS & CUDA-X
> Predefined kernels for data science

---

# CUDA-X & RAPIDS: kernels you don't have to write

<div class="cards">
  <div class="card"><h3>CUDA-X</h3><p>NVIDIA's library collection on top of CUDA: math, deep learning (cuDNN, TensorRT), comms, data science</p></div>
  <div class="card"><h3>RAPIDS</h3><p>The <b>data-science slice</b> of CUDA-X, open-source, with familiar Python APIs</p></div>
</div>

RAPIDS packages thousands of **optimized GPU kernels** behind APIs you already know:

| You already know | RAPIDS gives you | for |
|---|---|---|
| `pandas` · `scikit-learn` · `NumPy` · `NetworkX` | **cuDF** · **cuML** · **CuPy** · **cuGraph** | dataframes · ML · arrays · graphs |

---

# You stand on thousands of tuned kernels

<div class="kernwrap">
<div class="kernels">
<span>sort</span><span>join</span><span>group-by</span><span>filter</span><span>matmul</span><span>FFT</span><span>reduce</span><span>scan</span><span>k-means</span><span>PCA</span><span>random</span><span>SVD</span><span>rolling</span><span>histogram</span>
</div>
<div class="bigarrow">→</div>
<div class="callbox">one Python call<small>cuDF · cuML · CuPy</small></div>
</div>

<br>

- Each tile is a **GPU kernel** that experts wrote and tuned over years
- RAPIDS bundles them behind the **APIs you already know**
- You write familiar Python; **RAPIDS runs the CUDA for you**

> *How much* faster on real data? We'll explore that in this tutorial

---

<!-- _class: stackslide -->

# Putting it all together

<div class="stack">
  <div class="layer">Your code &amp; notebooks <small>pandas, scikit-learn, NumPy, your scripts</small></div>
  <div class="layer">RAPIDS and CUDA-X <small>cuDF · cuML · cuGraph, the data-science slice of CUDA-X</small></div>
  <div class="layer">CUDA Python <small>cuda.core · cuda.bindings · Numba · CuPy</small></div>
  <div class="layer">CUDA Toolkit <small>cuBLAS · cuSPARSE · NCCL · libcudart · nvcc</small></div>
  <div class="layer host">NVIDIA driver <small>libcuda, the system layer that talks to the hardware</small></div>
  <div class="layer hw">GPU hardware <small>SMs · warps · thousands of cores · VRAM</small></div>
</div>

> Each layer builds on the one below it; your Python sits at the very top.

---

# The mental model for today

- A GPU is **many simple cores + its own fast memory** → built for **parallel, math-heavy** work
- Getting data **onto** the GPU is the slow part → **keep it there**
- **CUDA** (C-based) exposes that power, but you rarely touch it
- **CUDA Python** and the **CUDA Toolkit** are the building blocks
- **RAPIDS / CUDA-X** hand you **ready-made kernels**, so familiar Python code just runs on the GPU

---

<!-- _class: title -->
<!-- _paginate: false -->

# Next: let's get on a GPU
