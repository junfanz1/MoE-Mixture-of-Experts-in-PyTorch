
<!-- TOC --><a name="mixture-of-experts-moe-implementation-in-pytorch"></a>
# Mixture-of-Experts (MoE) Implementation in PyTorch

This repository provides two implementations of a Mixture-of-Experts (MoE) architecture designed for research on large language models (LLMs) and scalable neural network designs. One implementation targets a **single-device/NPU environment** while the other is built for **multi-device distributed computing**. Both versions showcase the core principles of MoE architectures, including dynamic routing, expert specialization, load balancing, and capacity control.

## Content 

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [1. Overview and Purpose](#1-overview-and-purpose)
- [2. Architecture Components](#2-architecture-components)
- [3. Tech Stack](#3-tech-stack)
- [4. Distributed Computing and Load Balance Considerations](#4-distributed-computing-and-load-balance-considerations)
   * [4.1 Distributed Implementation (`moe_intral.py`)](#41-distributed-implementation-moe_intralpy)
   * [4.2 Single-Device Implementation (`moe_stand.py`)](#42-single-device-implementation-moe_standpy)
- [5. Breakdown Analysis: Functions and Model Lifecycle](#5-breakdown-analysis-functions-and-model-lifecycle)
   * [5.1 Expert Modules](#51-expert-modules)
   * [5.2 Data Generation and Preprocessing](#52-data-generation-and-preprocessing)
   * [5.3 Routing Mechanism](#53-routing-mechanism)
   * [5.4 Expert Assignment and Capacity Control](#54-expert-assignment-and-capacity-control)
   * [5.5 Distributed Communication and Load Balancing](#55-distributed-communication-and-load-balancing)
   * [5.6 Training and Inference Lifecycle](#56-training-and-inference-lifecycle)
- [6. Summary](#6-summary)
- [7. How to Use](#7-how-to-use)

<!-- TOC end -->

---

<!-- TOC --><a name="1-overview-and-purpose"></a>
# 1. Overview and Purpose

The primary goal of this implementation is to explore and experiment with MoE architectures that:
- **Enhance model capacity** by combining multiple expert networks.
- **Dynamically route** input tokens to the most relevant experts using top-k selection.
- **Maintain load balance** across experts via auxiliary loss functions.
- **Scale up** using distributed training (with PyTorch’s Distributed Data Parallel) across multiple devices (GPUs or NPUs).

These implementations are intended for research in large-scale model design, efficiency improvements in training, and investigating the interplay between expert specialization and routing strategies.

---

<!-- TOC --><a name="2-architecture-components"></a>
# 2. Architecture Components

1. **Expert Networks**
   - Each expert is a small feed-forward neural network (MLP) with a couple of linear layers and a GELU activation.
   - In the distributed version (`moe_intral.py`), experts are allocated to specific devices (e.g., `cuda:0`, `cuda:1`, …) for parallel execution.
   - The standard version (`moe_stand.py`) creates a list of expert modules that process input on a single device (or a specified NPU).

2. **Routing / Gating Mechanism**
   - A dedicated routing layer (or gate) computes logits from the input.
   - A softmax over these logits produces probability distributions indicating the “suitability” of each expert for the given input.
   - The **top-k** experts are selected per input sample, with corresponding routing probabilities used later for weighted aggregation.

3. **Capacity Control and Load Balancing**
   - **Capacity Factor:** Determines the maximum number of tokens each expert can process in a mini-batch.
   - **Load Balance Loss:** In both implementations, auxiliary losses are computed to balance expert usage:
     - In `moe_intral.py`, the loss is computed based on the difference between the mean router probability (density) and the actual expert usage aggregated across devices (using `all_reduce`).
     - In `moe_stand.py`, auxiliary losses include:
       - **Importance Loss:** Based on the variance of the aggregated routing probabilities.
       - **Load Balance Loss:** Derived from the average expert usage and routing weights.
   - These losses encourage the router to distribute the workload evenly, avoiding over-specialization and ensuring each expert contributes to the model output.

4. **Dynamic Expert Selection and Aggregation**
   - **Top-K Selection:** For each input sample, the router selects the top-k experts along with their weights.
   - **Weighted Aggregation:** Outputs from selected experts are weighted by the corresponding routing probabilities and then aggregated to reconstruct the final output.

---

<!-- TOC --><a name="3-tech-stack"></a>
# 3. Tech Stack

- **PyTorch:** Core deep learning framework used for model building, training, and distributed computing.
- **torch.distributed & DistributedDataParallel (DDP):** Employed in the distributed implementation for synchronizing gradients and expert usage across multiple processes/devices.
- **torch_npu:** Provides support for NPU devices, with utilities to automatically handle device transfers.
- **NCCL:** The backend used for multi-GPU communications in distributed training (via `torch.distributed.init_process_group`).

---

<!-- TOC --><a name="4-distributed-computing-and-load-balance-considerations"></a>
# 4. Distributed Computing and Load Balance Considerations

<!-- TOC --><a name="41-distributed-implementation-moe_intralpy"></a>
## 4.1 Distributed Implementation (`moe_intral.py`)
- **Device Allocation:** Each expert is explicitly assigned to a different GPU (using `cuda: {i}`) to enable parallel computation.
- **Distributed Data Sampling:** Uses `DistributedSampler` to ensure that data is partitioned evenly across devices.
- **Process Group Initialization:** The training routine initializes a distributed process group (using `dist.init_process_group` with the NCCL backend) and assigns each process a unique rank.
- **Global Aggregation:**
  - **Expert Usage:** The expert selection counts are aggregated across all processes using `dist.all_reduce` to obtain a global view.
  - **Auxiliary Losses:** Load balance and importance losses are computed on the aggregated data, ensuring that the routing mechanism is globally optimized.
- **Inter-Device Communication:** Inputs and expert outputs are explicitly transferred between devices to ensure that the correct data is processed by the corresponding expert and that outputs are aggregated back on the originating device.

<!-- TOC --><a name="42-single-device-implementation-moe_standpy"></a>
## 4.2 Single-Device Implementation (`moe_stand.py`)
- **Simplified Routing and Aggregation:** Runs entirely on one device, making it easier to debug and benchmark.
- **Capacity Control:** Each expert can process a limited number of tokens per batch (enforced via the `expert_capacity` parameter), which is crucial when simulating MoE behavior in resource-constrained environments.
- **Auxiliary Loss Computation:** Although less complex than the distributed version, the single-device version still implements balancing strategies to ensure no expert is under- or over-utilized during training.

---

<!-- TOC --><a name="5-breakdown-analysis-functions-and-model-lifecycle"></a>
# 5. Breakdown Analysis: Functions and Model Lifecycle

<!-- TOC --><a name="51-expert-modules"></a>
## 5.1 Expert Modules
- **`Expert` Class (both files):**
  - Implements a feed-forward network with linear layers and activation functions.
  - Provides the basic building block for expert specialization.

<!-- TOC --><a name="52-data-generation-and-preprocessing"></a>
## 5.2 Data Generation and Preprocessing
- **`gen_data` (in `moe_intral.py`):**
  - Simulates input data using a Gaussian distribution.
  - Generates random labels for testing and debugging the MoE training loop.

<!-- TOC --><a name="53-routing-mechanism"></a>
## 5.3 Routing Mechanism
- **`router` / `gate`:**
  - In `moe_intral.py`, a single linear layer computes logits which are then softmaxed to produce routing probabilities.
  - In `moe_stand.py`, a two-layer routing mechanism (a linear layer followed by softmax) performs a similar role.

<!-- TOC --><a name="54-expert-assignment-and-capacity-control"></a>
## 5.4 Expert Assignment and Capacity Control
- **Top-k Selection:**
  - Both implementations use `torch.topk` to select the highest scoring experts per sample.
  - This dynamic selection is key to the MoE strategy, ensuring that each sample is processed by the most relevant experts.
- **Capacity Enforcement:**
  - Limits the number of tokens processed by each expert, ensuring that the computational load is balanced.
  - In the distributed version, capacity is computed relative to the total batch size and the number of devices.

<!-- TOC --><a name="55-distributed-communication-and-load-balancing"></a>
## 5.5 Distributed Communication and Load Balancing
- **Global Aggregation (Distributed Version):**
  - Uses `dist.all_reduce` to aggregate expert usage counts and importance metrics.
  - The balance loss is computed by comparing expected expert usage with actual distribution across all devices.
- **Auxiliary Losses:**
  - Both implementations compute additional losses (importance loss and load balance loss) to steer the routing decisions during training.
  - These losses are integrated with the primary loss function to guide model optimization.

<!-- TOC --><a name="56-training-and-inference-lifecycle"></a>
## 5.6 Training and Inference Lifecycle
- **Initialization:**
  - **Distributed Training:** In `moe_intral.py`, processes are spawned using `torch.multiprocessing.spawn`, each initializing its process group and setting the device context.
  - **Standard Training:** In `moe_stand.py`, the model and data are loaded onto a single device (or NPU) for iterative training.
- **Training Loop:**
  - Data is loaded (with a distributed sampler in the distributed version), and the model processes batches of inputs.
  - The outputs are aggregated from expert computations, and the total loss (main loss + weighted auxiliary loss) is computed.
  - Backpropagation updates model parameters across the experts and the routing network.
- **Evaluation:**
  - After training, the model switches to evaluation mode (`model.eval()`) to assess inference performance without auxiliary loss interference.

---

<!-- TOC --><a name="6-summary"></a>
# 6. Summary

This repository demonstrates two approaches to implementing Mixture-of-Experts:
- **Distributed MoE (`moe_intral.py`):** Emphasizes multi-device training, inter-device communication, and global load balancing for large-scale systems.
- **Standard MoE (`moe_stand.py`):** Provides a simpler, single-device variant that is useful for preliminary experiments and debugging.

Both implementations highlight key challenges in MoE research such as dynamic routing, expert capacity constraints, and load balancing. They serve as a robust starting point for further exploration in scalable and efficient deep learning model designs.

---

<!-- TOC --><a name="7-how-to-use"></a>
# 7. How to Use

1. **Distributed Version (`moe_intral.py`):**
   - Set up the necessary environment variables for distributed training (`MASTER_ADDR` and `MASTER_PORT`).
   - Launch the training using PyTorch’s multiprocessing support:
     ```bash
     python moe_intral.py
     ```
   - Ensure that the system has 8 available devices (GPUs/NPUs).

2. **Standard Version (`moe_stand.py`):**
   - Run directly on a single device (or NPU):
     ```bash
     python moe_stand.py
     ```
   - Adjust hyperparameters and device configurations as needed.

---

## Acknowledgements

[Zomi's AI Infra Github](https://github.com/chenzomi12/AIInfra)

## License

Distributed under the MIT License. See the [LICENSE](LICENSE) file for more information.
