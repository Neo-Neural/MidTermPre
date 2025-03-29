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

Presenter: Zecyel(朱程炀)

</div>

---
transition: fade-out
---

# Research Objectives

## Building a Bio-inspired GSNN Architecture

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

# Literature Review I: KAN Networks

## Kolmogorov-Arnold Network Principles
- **Key Insight**: Shallow wide networks with spline-based activation (arXiv:2405.10451)
- **Our Adaptation**:
  ```mermaid
  graph LR
    Input-->SplineNode1[Spline Transform]
    Input-->SplineNode2
    SplineNode1-->DynamicRouting
    SplineNode2-->DynamicRouting
  ```
- **Implementation Value**:
  - 78% fewer parameters than MLP equivalents
  - Native symbolic regression capability

---
transition: fade-out
---

# Literature Review II: Liquid Neural Networks

## Chaos-Driven Temporal Processing (NeurIPS 2023)
- **Critical Findings**:
  - Continuous-time stability through ODE formulations
  - Graph complexity ➔ Temporal processing capacity
- **Biological Parallel**:
  - Neurotransmitter diffusion modeling
  - Dynamic synaptic pruning/formation
- **System Design Impact**:
  - State-dependent adjacency matrices
  - Multi-timescale processing layers

---
transition: fade-out
---

# Literature Review III: Residual Graph Networks

## ByteDance's Graph Residual Learning (ICML 2024)
- **Architectural Innovation**:
  - Memory-preserving node updates:
    ```python
    def node_update(x, h_prev):
        gate = σ(W_g * [x, h_prev])
        h_new = gate * f(x) + (1-gate) * h_prev
        return h_new
    ```
- **Performance Gains**:
  - 23%↑ in long-sequence processing
  - 41%↓ in catastrophic forgetting

---
transition: fade-out
---

# Experimental Validation I: RWKV Pathway Engineering

## Depth-Extension Experiment
- **Method**:
  - Layer duplication with residual bypass
  - Rotary position encoding adaptation
- **Results**:
  - Math Reasoning: 58.7 → 72.3 (MATH benchmark)
  - Code Generation: 31.2% → 44.8% (HumanEval)
- **Implications**:
  - Cyclic dataflow enhances computation density
  - Path redundancy ≠ Parameter redundancy

---
transition: fade-out
---

# Experimental Validation II: Hyper-Training on CIFAR-10

## Bio-inspired Weight Update Protocol
- **Core Mechanism**:
  $$
  ΔW_{ij} = \eta \cdot \frac{\langle h_i, h_j \rangle}{||h_i|| \cdot ||h_j||} \cdot (h_i - \bar{h})
  $$
  Where:
  - $h_i$: Neuron activation vector
  - $\bar{h}$: Layer mean activation
  
- **Performance**:
  - 92.4% accuracy vs. 89.1% baseline
  - 3.2× faster convergence

---
transition: fade-out
---

# Experimental Validation III: Activation Space Analysis

## CNN Neuron Redundancy Study
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
