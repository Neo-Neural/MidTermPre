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

# Table of Content

1. Fundamental Challenges in Current LLMs
2. Research Objectives
3. Literature Review I: Kolmogorov-Arnold Networks
4. Literature Review II: Liquid Neural Networks
5. Literature Review II: Hyper-Connections Network
6. Experimental Validation I: RWKV Pathway Engineering
7. Experimental Validation II: Hyper-Training on CIFAR
8. Experimental Validation III: CNN Redundancy Analysis
9. Future Roadmap


---
transition: fade-out
---

# Fundamental Challenges in Current LLMs

## Architectural Limitations in Modern Foundation Models

<div h-5 />

<div flex flex-auto space-x-4 flex-nowarp>
<div max-w-110>

1. **Tool Utilization Deficiency**
   - Static API binding protocols
   - No dynamic tool composition ability
   - 72% failure rate in multi-step tool chaining (Google, 2024)

2. **Multimodal Integration Barriers**
   - Modality projection bottlenecks
   - Cross-modal attention collapse
   - Information loss during fusion

</div>
<div>

3. **Knowledge Representation Crisis**
   - Parametric memorization inefficiency
   - 1 parameter ≈ 2bit information (Physics of LLMs theory)
   - Catastrophic forgetting during updates

4. **Structural Rigidity**
   - Fixed topological organization
   - No self-modification capability

</div>
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

![rwkv-mmlu](./assets/rwkv-mmlu.jpg){.w95}
![rwkv-arc](./assets/rwkv-arc.png){.w95}
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

# Experimental Validation II: Hyper-Training on CIFAR

## Hyperparameter Optimization

<div flex flex-auto space-x-25 flex-nowarp>
<div max-w-110>

<div h-5 />

- **Key Findings**:
  - **Sweet Spot**: batch_size=40, learning_rate=0.1

- **Biological Interpretation**:
  - Smaller batches mimic biological "mini-batches"
  - High LR enables rapid synaptic plasticity

</div>
<div>

![hyper-parameter](./assets/hyper-parameter.png){.w70}

</div>
</div>

---
transition: fade-out
---

# Experimental Validation III: CNN Redundancy Analysis

## PCA Visualization of CNN Neuron Space

<div flex flex-auto space-x-25 flex-nowarp>
<div max-w-110>

<div h-10 />

1. Pretrained CNN
2. Forward & Record Activation
3. Get Batch×Neuron Matrix
4. Unit Normalization
5. Concatenate Inverted Vectors
6. PCA Transformation
7. Redundancy Quantification

</div>
<div>

![pca](./assets/pca.png){.w120}

</div>
</div>

---
transition: fade-out
---

# Future Roadmap

## Towards Graph Neural Architectures

<div h-5 />

<div flex flex-auto space-x-10 flex-nowarp>
<div max-w-115>

1. **Multimodal Expansion**  
   - Hidden state augmentation for speech-text fusion  
   - Cross-modal attention through shared graph space  

2. **Tool-Oriented LLM**  
   - Dynamic API binding via graph edges  
   - Self-discovered tool composition patterns  

</div>
<div>

3. **Macroscopic Graph Networks**  
   - Vector neurons with tensor edges  
   - Emergent subgraph specialization  

4. **Ablation Roadmap**  
   - Phase 1: Isolated component validation
   - Phase 2: Pairwise integration tests
   - Phase 3: Full system optimization
</div>
</div>

---
class: text-center font-size-20 mt-45
---

Thanks for listening.
