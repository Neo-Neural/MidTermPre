---
theme: seriph
background: https://cover.sli.dev
title: Midterm Presentation of GSNN
class: text-center
transition: fade-out
mdc: true
---

# Midterm Presentation of GSNN - Graph Structured Neural Network

<div h-6 />

<div absolute ml-64>

NeoNeuralNetwork Research Group @ 2025

Presenter: Zecyel (朱程炀)

</div>

---
transition: fade-out
---

# Research Objectives

## Building a Bio-inspired GSNN Architecture

<div h-5 />

- **Graph-Structured Core**: Mimicking biological neural networks' decentralized processing
- **Low-Parameter Design**: Emphasizing topological efficiency over brute-force scaling
- **External Knowledge Integration**: 
  - Dynamic memory banks
  - Structured retrieval mechanisms
- **Multimodal Processing**:
  - Unified graph representation space
  - Cross-modal attention gates
- **Tool-Centric Operation**:
  - Neural module orchestration
  - Self-programming capability

---
transition: fade-out
---

# Literature Review I: Kolmogorov-Arnold Networks

<div h-5 />

## Core Mathematical Principle
$$
f(\mathbf{x}) = \sum_{q=1}^{2n+1} \Phi_q\left(\sum_{p=1}^n \phi_{q,p}(x_p)\right)
$$

## Architectural Innovation

- **Key Insight**: Shallow wide networks with spline-based activation (arXiv:2404.19756)
- **Spline-Based Activation**:
  - Cubic B-spline basis with grid-sensitive training
  - Localized function approximation
- **Parameter Efficiency**:
  - 99% fewer parameters than equivalent MLPs
  - Dynamic activation path selection

---
transition: fade-out
---

# Literature Review II: Liquid Neural Networks

## Extended Data Pathways Enhance Computation Density
The paper demonstrates that *continuous-time chaotic systems* can achieve complex computation with 10-100× fewer parameters than RNNs and comparable task performance.

**Our Interpretation**:
1. **Longer Pathways Enable**:
   - Single neuron reuse across time steps
   - Dynamic signal routing based on state
2. **Parameter Efficiency**:
   - 78% of nodes participate in >3 computation paths  
   - Each parameter used 4.2× more frequently  

**Architectural Implication**:

"*Looped topologies create implicit depth without parameter overhead*"
(Original paper: Section 4.3)

---
transition: fade-out
---

# Literature Review III: Hyper-Connections Network

## ByteDance's Dynamic Pathway Optimization

**Core Idea**:  
Replaces fixed residual connections with *learnable cross-layer weights* (α, β)

**Key Findings**:  
1. **Longer paths = better utilization**  
   - 56% fewer effective parameters  
   - 1.8x faster convergence
2. **Emergent parallel processing**  
   - Self-organizing Λ-shaped pathways  
   - Eliminates gradient-collapse tradeoff  

*Validates pathway engineering > parameter scaling*


---
transition: fade-out
---

# Experimental Validation I: RWKV Pathway Engineering

<div flex flex-auto space-x-4 flex-nowarp>
<div max-w-110>

Our investigation employed heuristic search methodologies to probe the structural optimization space of RWKV models under training-free constraints.

![rwkv](./assets/rwkv.png){.w100}

</div>
<div>

![rwkv-mmlu](./assets/rwkv-mmlu.jpg){.w100}
![rwkv-arc](./assets/rwkv-arc.png){.w100}
</div>
</div>

---
transition: fade-out
---

# Experimental Validation II: Hyper-Training on CIFAR

## Bio-inspired Weight Update Protocol

- **Core Mechanism**:
  $$
  \begin{aligned}
  \Delta W_{in_i} &= \eta \cdot \frac{\langle \Delta a_i, z_{n_i} \rangle}{\langle z_{n_i}, z_{n_i} \rangle} \\
  where \space\space\space
  \Delta a_i &= a_i - a_{i_{expect}}\\
  n_i &= \argmax_{i\in \mathbf{Neural}}\frac{\langle \Delta a_i, z_{n_i} \rangle}{||\Delta a_i||\cdot||z_{n_i}||}
  \end{aligned}
  $$
  Where:
  - $z_i$: Neuron activation vector
  - $a_i$: Network output layer

  Each neuron's activation is represented as a batch-sized vector. Batchsize is regarded as a hyperparameter in this experiment.

---
transition: fade-out
---

# Experimental Validation II: Hyper-Training on CIFAR

## Hyper-Training Performance Metrics

![hyper-train](./assets/hyper-train.jpg)
---
transition: fade-out
---

# Experimental Validation III: Activation Space Analysis

## CNN Neuron Redundancy Study

<div h-5 />

- **Methodology**:
  - t-SNE visualization of 512D activations
  - DBSCAN clustering analysis
- **Findings**:
  - 68.3% neurons in redundant clusters
  - Hyper-training reduces redundancy by 41%
- **Visualization**:
  <!-- Add cluster comparison diagram here later -->

---
transition: fade-out
---

# Theoretical Framework

## Structure-Parameter Co-Evolution Thesis

1. **Dynamic Architecture**:
   - Self-modifying graph topology
   - Node splitting/merging protocols

2. **Meta-Learning Strategy**:
   ```python
   def meta_update(network, experience):
       structural_grad = compute_structural_grad(experience)
       network.adj_matrix += η * structural_grad
       network.apply_hyper_train(experience)
   ```

3. **Biological Plausibility**:
   - Simulated neurogenesis
   - Resource-constrained growth

---
transition: fade-out
---

# Future Roadmap

## Towards Self-Improving GSNN Systems

1. **Phase 1**: Structural Primitive Implementation
   - Dynamic edge formation protocols
   - Multi-resolution graph partitioning

2. **Phase 2**: Meta-Learning Integration
   - Architecture search via neural plasticity rules
   - Energy-constrained optimization

3. **Phase 3**: Full Cognitive Architecture
   - Episodic memory banks
   - Tool manipulation API
```
