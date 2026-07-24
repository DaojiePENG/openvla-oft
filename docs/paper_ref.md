# Latency-Robust Vision-Language-Action Models via Cloud-Edge Collaborative Inference

> Technical reference document for AAAI 2026 submission.
> Last updated: 2026-07-23

---

## Abstract

Vision-Language-Action (VLA) models have emerged as powerful generalist robot policies by unifying high-level semantic reasoning with low-level motor control within a single large-scale backbone. However, deploying such models under realistic network conditions introduces a fundamental tension: the computationally intensive backbone that provides rich planning representations must typically run on a remote server with non-negligible latency, while the robot's local controller demands real-time responsiveness. Existing dual-system approaches attempt to address this through action-chunk re-planning or strictly synchronized fast-slow pipelines, but these solutions impose rigid temporal alignment constraints—requiring the fast system to operate at a fixed multiple of the slow system's frequency—and degrade gracefully only within narrow operating regimes.

We propose a fundamentally different paradigm: a **latency-tolerant cloud-edge collaborative VLA framework** that eliminates the need for temporal alignment between the cloud and edge. Our key insight is that the two subsystems should learn **complementary and temporally asymmetric representations**: the cloud-side VLA backbone learns **time-invariant high-level representations** encoding task semantics, goals, and strategic trends (latency-insensitive), while the edge-side lightweight action head learns to **ground these high-level decisions into precise motor commands** using real-time local vision (latency-sensitive). This representational separation is achieved through a **paired-frame dual-path training strategy** that naturally encourages the backbone to discard timing-critical information and the action head to compensate with real-time visual features. On LIBERO-Goal, CloudEdgeVLA reaches 96.0% success without delay and 94.2% at a 40-step delay (98.1% retention), while the strongest competing baseline reaches only 3.0% at the same delay. A matched-checkpoint diagnostic further shows 32.9% lower fresh–stale action drift than single-frame training at a 20-step delay, supporting the intended functional robustness of the learned fusion policy.

---

## 1. Introduction

### 1.1 Motivation

The rapid scaling of Vision-Language-Action (VLA) models has demonstrated that large vision-language backbones can serve as effective generalist robot policies. Models such as RT-2, OpenVLA, and π₀ leverage billions of parameters and internet-scale pretraining to achieve strong semantic reasoning over visual observations and natural language instructions. However, this capability comes at a significant computational cost: inference on a 7B-parameter VLA backbone typically requires high-end GPU hardware that cannot be practically mounted on mobile robot platforms.

This motivates a **cloud-edge split** architecture, where the computationally intensive backbone runs on a remote server (cloud), while a lightweight action head runs locally on the robot (edge). The cloud processes observations and produces high-level planning features; the edge fuses these features with real-time sensor data to produce actions at control frequency.

The critical challenge is **network latency**. In a typical deployment, the cloud-side backbone processes observations that are delayed by the communication round-trip time (RTT). If the edge action head naively consumes these stale planning features without accounting for the temporal gap, the resulting actions are misaligned with the current state of the environment, leading to task failures.

### 1.2 Limitations of Existing Dual-System Approaches

Prior work on dual-system robot architectures has explored two main strategies for handling the interface between a slow deliberative system and a fast reactive system:

1. **Action-chunk re-planning**: The slow system predicts a chunk of $T$ future actions, and the fast system executes them open-loop before requesting a new plan. This approach assumes that the environment remains approximately static during the chunk execution—an assumption that is violated when the slow system's observations are delayed, compounding the staleness of the plan.

2. **Synchronized fast-slow pipelines**: The fast system runs at a fixed multiple $\alpha$ of the slow system's frequency (e.g., $\alpha = 5\times$), consuming intermediate representations from the slow system at each fast tick. This requires **strict temporal alignment**: the fast system must know exactly when the slow system will produce new features, and the slow system's latency must be bounded and predictable—conditions that are rarely met in real-world networked deployments.

Both approaches share a common limitation: they treat the slow system's output as a **precise, time-critical signal** that must be consumed at the right moment. This creates a fragile coupling between the two systems, where any deviation in timing (network jitter, variable inference latency, load fluctuations) directly degrades performance.

### 1.3 Our Paradigm: Temporally Asymmetric Representation Learning

We propose a fundamentally different approach that **eliminates the temporal alignment requirement** entirely. Our framework is built on a key insight from the cognitive science literature on dual-process theory: in a well-designed dual-system architecture, the two subsystems should operate on **different temporal scales and represent different levels of abstraction**.

We draw an analogy to Kahneman's System 1 / System 2 framework:

| | System 2 (Cloud VLA) | System 1 (Edge Action Head) |
|:--|:--|:--|
| **Role** | High-level planning, goal reasoning, semantic understanding | Low-level reactive control, precise motor execution |
| **Temporal sensitivity** | **Latency-insensitive**: trends and decisions change slowly | **Latency-sensitive**: must respond to immediate environmental changes |
| **Representation** | Time-invariant task semantics, goals, strategies | Time-critical spatial details, contact states, perturbations |
| **Update rate** | Can tolerate seconds of delay | Must operate at control frequency (10–50 Hz) |
| **Compute** | Large backbone (7B+ params) | Lightweight head (<1% of backbone) |

The critical design principle is: **each system should learn representations that match its temporal role**. The cloud backbone should produce features that encode *what to do* (task goals, manipulation strategy, object semantics) rather than *exactly when to do it* (precise timing, instantaneous contact forces). The edge action head should combine these high-level directives with real-time vision to determine *how to do it right now*.

This separation is not manually engineered—it **emerges naturally** from our paired-frame training strategy. By forcing the backbone to produce features that support correct action prediction from both fresh and delayed observations, the training objective implicitly encourages the backbone to discard timing-critical information and retain time-invariant semantic information. Conversely, the action head learns to rely on its local vision encoder for timing-critical details.

### 1.4 Problem Statement

We formalize the asynchronous cloud-edge VLA deployment problem as follows. At each environment step $t$:

- The **edge** captures the current observation $o_t$ and sends it to the cloud.
- Due to network latency, the cloud receives $o_{t-k}$ (delayed by $k$ steps) and produces planning features $h_{t-k} = f_\theta(o_{t-k}, \ell)$, where $\ell$ is the language instruction and $f_\theta$ is the VLA backbone.
- The **edge** must produce the action $a_t$ using the stale planning features $h_{t-k}$ and its own real-time observation $o_t$.

The goal is to learn an action head $g_\phi$ such that $a_t = g_\phi(h_{t-k}, v_\psi(o_t))$ achieves high task success despite the temporal mismatch between $h_{t-k}$ and $o_t$, where $v_\psi$ is a local vision encoder.

### 1.5 Contributions

1. **Latency-Tolerant Cloud-Edge VLA Framework**: A dual-system architecture that decouples high-level planning (cloud, latency-insensitive) from low-level reactive control (edge, latency-sensitive), with no temporal alignment requirement between the two systems.

2. **Vision-Augmented Action Head (VAH)**: A lightweight edge-side module that fuses stale cloud planning features with real-time local vision features, enabling the edge to compensate for arbitrary delays using its own visual grounding.

3. **Paired-Frame Training Strategy**: A training procedure that exposes the action head to both fresh and stale planning features within each training step, implicitly encouraging **representational specialization**: the backbone learns time-invariant semantics, while the action head learns time-critical visual-motor control.

4. **Paradigm Shift from Existing Dual-System Approaches**: Unlike action-chunk re-planning or synchronized fast-slow pipelines, our approach requires neither bounded delay, nor temporal alignment, nor a fixed frequency ratio between cloud and edge. The edge simply consumes the most recently available planning features and compensates for their staleness using real-time vision.

---

## 2. Related Work

### 2.1 Vision-Language-Action Models

Recent VLA models (RT-2, Octo, OpenVLA, π₀, OpenVLA-OFT) demonstrate that large vision-language backbones can serve as generalist robot policies. These models typically process visual observations through a vision encoder, condition on language instructions via a language model backbone, and predict actions either as discretized tokens or continuous regression targets. Our work builds on this paradigm but addresses the previously unexplored problem of deploying such models under asynchronous network conditions.

### 2.2 Cloud-Edge Robot Architectures

Cloud robotics has a rich history, from early work on cloud-based SLAM and grasping to more recent approaches that offload perception and planning to remote servers. However, most prior work treats the cloud as a pure planning module and the edge as a pure execution module, without addressing the representational mismatch caused by latency. Our work bridges this gap by designing the action head to explicitly compensate for stale planning features.

### 2.3 Latency Robustness in Control

Robustness to delayed observations has been studied in the model-based control and reinforcement learning literature, typically through state augmentation (stacking delayed observations) or learned world models. Our approach differs fundamentally: we do not attempt to reconstruct the current state from delayed observations. Instead, we leverage the edge's real-time vision to directly correct the stale planning signal, which is both simpler and more effective for the VLA setting.

### 2.4 Dual-System Robot Architectures: A Critical Comparison

Several recent works adopt a dual-system or hierarchical architecture for robot control:

| Approach | Slow System | Fast System | Temporal Coupling | Delay Handling |
|:--|:--|:--|:--|:--|
| **RT-2 + diffusion policy** | VLA backbone (1–3 Hz) | Diffusion action model (10 Hz) | Strict: fast system consumes slow system's output at fixed intervals | Not addressed |
| **π₀** | VLM backbone | Flow-matching action head | Integrated: single forward pass | Not addressed (assumes sync) |
| **SayCan / Inner Monologue** | LLM planner | Skill library | Strict: replan triggered by skill completion | Not addressed |
| **Dual-system IL** | Slow policy (low freq) | Fast policy (high freq, $\alpha \times$) | **Strict $\alpha$ frequency ratio** | Bounded delay only |
| **Ours** | Cloud VLA backbone | Edge VisionActionHead | **None**: asynchronous, no fixed ratio | Handled by design |

The key differentiator of our approach is the **absence of temporal coupling**. Existing dual-system approaches require the fast system to consume the slow system's output at precisely defined intervals. When the slow system's latency varies (as it inevitably does in networked deployments), these approaches degrade because the fast system's training distribution (fixed $\alpha$ ratio) does not match the deployment distribution (variable latency).

Our framework avoids this problem entirely: the edge action head does not need to know *when* the cloud features were produced. It simply fuses the most recently received features with real-time vision and predicts actions. The paired-frame training ensures that the action head is robust to any delay within the training range.

---

## 3. Method

### 3.1 System Architecture

Our framework consists of three components operating under a clear **temporal asymmetry**:

**Cloud-Side VLA Backbone** $f_\theta$ (System 2, latency-insensitive): A large-scale vision-language model that processes visual observations and language instructions to produce high-level planning representations. Given an observation $o$ and instruction $\ell$, the backbone produces a sequence of hidden-state representations:

$$h = f_\theta(o, \ell) \in \mathbb{R}^{L \times D}$$

where $L$ is the number of action-token positions (corresponding to an action chunk of length $T$ with $A$ action dimensions, so $L = T \times A$) and $D$ is the hidden dimension. The backbone is parameterized by LoRA-adapted weights on top of a pretrained vision-language model and runs exclusively on the cloud server. Importantly, the backbone is designed to produce features that encode **what** to do (task goals, manipulation strategy, object semantics) rather than **exactly when** to do it (see Section 3.7 for analysis).

**Edge-Side Vision Encoder** $v_\psi$ (System 1 perception, latency-sensitive): A lightweight vision encoder (e.g., SigLIP-Base) that runs locally on the robot and extracts real-time visual features from the current observation:

$$z_t = v_\psi(o_t) \in \mathbb{R}^{D_v}$$

The vision encoder is frozen during training to leverage pretrained visual representations and reduce the edge-side computational footprint. Crucially, $z_t$ is always computed from the **current** observation $o_t$, making it a delay-free signal that captures the instantaneous state of the environment.

**Edge-Side Action Head** $g_\phi$ (System 1 control, latency-sensitive): A learnable module that fuses the (potentially stale) cloud planning features with the real-time edge vision features to predict continuous actions:

$$\hat{a}_t = g_\phi(h_{t-k}, z_t) \in \mathbb{R}^{T \times A}$$

The action head first mean-pools the planning features over the action-dimension axis to obtain per-timestep planning embeddings, projects the vision features into the same latent space, concatenates them, and passes the result through a residual MLP to predict the action chunk. The action head acts as a **real-time grounding layer**: it translates the cloud's high-level (but potentially stale) directives into precise motor commands using the edge's current visual context.

### 3.2 Dual-Path Training with Paired Frames

The key challenge in training the collaborative system is that the action head must learn to produce correct actions from **both** fresh and stale planning features. We achieve this through a paired-frame training strategy that leverages the temporal structure of demonstration trajectories.

#### 3.2.1 Paired Frame Extraction

During training, each sample from the demonstration dataset provides a window of $W$ consecutive observations from the same episode:

$$\mathcal{W}_t = \{o_{t-W+1}, o_{t-W+2}, \ldots, o_t\}$$

From this window, we construct two observations:
- **Current frame**: $o_t$ (the latest observation in the window)
- **Delayed frame**: $o_{t-d}$, where $d \sim \text{Uniform}(1, W-1)$ is a randomly sampled delay

This ensures that the delayed frame always comes from the **same episode** as the current frame, preserving semantic consistency.

#### 3.2.2 Dual Forward Pass

Both frames are processed through the cloud backbone to produce planning features:

$$h^{\text{fresh}} = f_\theta(o_t, \ell), \quad h^{\text{stale}} = f_\theta(o_{t-d}, \ell)$$

Note that $h^{\text{stale}}$ is **not detached** from the computation graph. Both the fresh and stale planning features receive gradients, enabling the backbone to learn representations that are informative even when derived from delayed observations.

#### 3.2.3 Vision-Augmented Action Prediction

Both sets of planning features are fused with the **same** real-time vision features $z_t = v_\psi(o_t)$:

$$\hat{a}^{\text{fresh}} = g_\phi(h^{\text{fresh}}, z_t), \quad \hat{a}^{\text{stale}} = g_\phi(h^{\text{stale}}, z_t)$$

#### 3.2.4 Dual-Path Loss

Both predictions are trained toward the same ground-truth action $a_t$:

$$\mathcal{L} = \underbrace{\|\hat{a}^{\text{fresh}} - a_t\|_1}_{\mathcal{L}_{\text{fresh}}} + \underbrace{\|\hat{a}^{\text{stale}} - a_t\|_1}_{\mathcal{L}_{\text{stale}}}$$

This dual-path objective has the following effects:

- **$\mathcal{L}_{\text{fresh}}$** trains the action head to produce correct actions from fresh planning features (normal synchronous operation). It also provides direct action supervision to the backbone.
- **$\mathcal{L}_{\text{stale}}$** trains the action head to **compensate** for stale planning features using real-time vision. When $h^{\text{stale}}$ is misaligned with the current state due to delay, the action head must rely more heavily on $z_t$ to recover the correct action.

### 3.3 Implicit Consistency Regularization

The dual-path loss implicitly enforces a **consistency property** between the action head's predictions under fresh and stale planning features. Consider the gradient of $\mathcal{L}_{\text{stale}}$ with respect to the action head parameters $\phi$:

$$\nabla_\phi \mathcal{L}_{\text{stale}} = \nabla_\phi \|g_\phi(h^{\text{stale}}, z_t) - a_t\|_1$$

This gradient pushes $g_\phi$ to produce the same action from $h^{\text{stale}}$ as it would from $h^{\text{fresh}}$, effectively learning a mapping:

$$g_\phi(h^{\text{stale}}, z_t) \approx g_\phi(h^{\text{fresh}}, z_t)$$

This means that for small delays $d$, the action head learns to use $z_t$ to "fill in" the information gap between $h^{\text{stale}}$ and $h^{\text{fresh}}$. For larger delays, the action head relies more heavily on $z_t$ and treats $h^{\text{stale}}$ as a coarse prior rather than a precise planning signal.

**Proposition 1** (Informal). *Under the dual-path loss, if the vision encoder $v_\psi$ captures sufficient task-relevant information, the action head $g_\phi$ achieves bounded action prediction error for delays $d$ up to the maximum delay seen during training, with the error bound decreasing as the mutual information $I(z_t; a_t \mid h^{\text{stale}})$ increases.*

### 3.4 Training Procedure

The complete training procedure is summarized as follows:

```
For each training step:
    1. Sample a demonstration window W_t = {o_{t-W+1}, ..., o_t}
    2. Sample delay d ~ Uniform(1, W-1)
    3. Set o_current = o_t, o_delayed = o_{t-d}
    4. h_fresh = f_θ(o_current, ℓ)          // cloud backbone on current frame
    5. h_stale = f_θ(o_delayed, ℓ)          // cloud backbone on delayed frame
    6. z_t = v_ψ(o_current)                 // edge vision on current frame
    7. a_fresh = g_φ(h_fresh, z_t)          // action head on fresh features
    8. a_stale = g_φ(h_stale, z_t)          // action head on stale features
    9. L = ||a_fresh - a_gt||_1 + ||a_stale - a_gt||_1
    10. Update θ, φ via ∇L                  // ψ frozen
```

**Computational cost**: Each training step requires two forward passes through the cloud backbone (one for $o_t$, one for $o_{t-d}$). This doubles the backbone compute per step compared to standard single-frame training, but does not increase the action head or vision encoder cost. In practice, the two backbone passes can be batched along the batch dimension to reduce wall-clock overhead.

### 3.5 Deployment Protocol

At test time, the system operates as follows:

```
For each environment step t:
    1. Edge captures observation o_t
    2. Edge sends o_t to cloud (asynchronous, non-blocking)
    3. When cloud returns h (possibly from a previous o_{t-k}):
         h_received = h
    4. Edge computes z_t = v_ψ(o_t)        // real-time, local
    5. Edge computes a_t = g_φ(h_received, z_t)
    6. Edge executes a_t
```

The cloud backbone operates asynchronously: it continuously processes the latest available observation and returns planning features. The edge action head always uses the most recently received planning features combined with the current real-time vision. This design ensures that the robot **never blocks** waiting for the cloud, maintaining real-time control responsiveness regardless of cloud latency.

### 3.6 Delay Generalization

A key property of our approach is that the action head trained with uniformly sampled delays generalizes to **unseen delay distributions** at test time. This is because:

1. The vision encoder provides a **delay-invariant** signal ($z_t$ depends only on $o_t$, not on the delay).
2. The planning features provide a **delay-dependent** signal that the action head learns to appropriately weight.
3. For any delay $d$ within the training range, the action head interpolates between "trust planning features" (small $d$) and "trust vision" (large $d$).

Empirically, we observe that training with a maximum delay of $W-1$ generalizes to delays up to and beyond $W-1$, though with gradually increasing action error as the delay exceeds the training distribution.

### 3.7 Emergent Representational Specialization

A central claim of our work is that the paired-frame dual-path training induces **representational specialization** between the cloud backbone and the edge action head, without any explicit regularization or architectural constraint to enforce it. We analyze this phenomenon theoretically and empirically.

#### 3.7.1 Why the Backbone Learns Time-Invariant Representations

Consider the gradient of $\mathcal{L}_{\text{stale}}$ with respect to the backbone parameters $\theta$:

$$\nabla_\theta \mathcal{L}_{\text{stale}} = \nabla_\theta \|g_\phi(f_\theta(o_{t-d}, \ell), v_\psi(o_t)) - a_t\|_1$$

This gradient backpropagates through the action head $g_\phi$ into the backbone $f_\theta$. Crucially, the supervision signal $a_t$ corresponds to the **current** time step $t$, but the backbone input is the **delayed** observation $o_{t-d}$. This creates a pressure for the backbone to produce features from $o_{t-d}$ that are still useful for predicting the action at time $t$.

The only way the backbone can consistently satisfy this pressure across **random delays** $d \sim \text{Uniform}(1, W-1)$ is to encode information that is **invariant to the delay**—that is, information about the task goal, manipulation strategy, and object semantics that remains valid regardless of which frame in the recent history is observed. Conversely, information that is timing-critical (e.g., the exact position of the gripper at time $t$, the instantaneous contact force) is **unreliable** from a delayed frame and therefore receives weaker gradient signal from $\mathcal{L}_{\text{stale}}$.

Formally, let $I_{\text{delay}}(h; a_t)$ denote the mutual information between the planning features $h$ and the target action $a_t$ that is attributable to delay-critical information, and $I_{\text{inv}}(h; a_t)$ denote the mutual information attributable to delay-invariant information. The dual-path loss satisfies:

$$\mathbb{E}_{d}[\mathcal{L}_{\text{stale}}] \propto -I_{\text{inv}}(h^{\text{stale}}; a_t) + \text{const}$$

while $I_{\text{delay}}$ contributes less to reducing $\mathcal{L}_{\text{stale}}$ as $d$ increases. Over training, this leads to:

$$I_{\text{inv}}(f_\theta(o, \ell); a_t) \gg I_{\text{delay}}(f_\theta(o, \ell); a_t)$$

i.e., the backbone's representations become dominated by time-invariant semantic information.

#### 3.7.2 Why the Action Head Learns Time-Critical Visual-Motor Control

The action head receives both the (stale) planning features and the (real-time) vision features. Through $\mathcal{L}_{\text{stale}}$, the action head learns that when the planning features are stale (i.e., less informative about the current state), it must rely more heavily on $z_t$ to produce the correct action. Through $\mathcal{L}_{\text{fresh}}$, it learns that when the planning features are fresh, $z_t$ serves as a complementary fine-grained signal.

Over training, the action head develops a **soft attention mechanism** between the planning features and the vision features:

$$g_\phi(h, z_t) \approx \alpha(d) \cdot \text{MLP}_h(h) + (1 - \alpha(d)) \cdot \text{MLP}_z(z_t) + \text{cross-interaction}$$

where $\alpha(d) \to 1$ for small delays (trust planning) and $\alpha(d) \to 0$ for large delays (trust vision). This is not a hard switch but a continuous interpolation learned end-to-end.

#### 3.7.3 Comparison with Standard VLA Training

In standard VLA training (single frame, no delay), the backbone receives direct action supervision from $\mathcal{L}_{\text{fresh}}$ and learns to encode **all** task-relevant information—including timing-critical details—into its hidden states. The action head is a simple projection from hidden states to actions, with no need for real-time visual grounding.

This creates a **fragile dependency**: if the backbone's observations are delayed even slightly, the timing-critical information in its hidden states becomes stale, and the action head has no mechanism to compensate. This is precisely why standard VLAs degrade sharply under latency.

In contrast, our training procedure creates a **robust division of labor**: the backbone handles what-to-do (delay-tolerant), and the action head handles how-to-do-it-now (delay-sensitive). This division makes the system inherently resilient to latency.

---

## 4. Experiments

The evidence below uses two complementary views of latency robustness. The closed-loop experiment measures the end-to-end outcome that matters for deployment: task success under delayed cloud features. The fresh–stale action-consistency diagnostic then isolates the policy interface by holding current edge vision fixed and changing only the age of the cloud planning feature. Together, they provide behavioral and functional evidence for the robustness targeted by dual-path training.

### 4.1 Implementation Framework

To validate our cloud-edge collaborative VLA framework, we implement the full training and evaluation pipeline on top of **OpenVLA-OFT**, a recent open-source VLA architecture that provides:

- A 7B-parameter vision-language backbone with LoRA fine-tuning support
- An action chunk prediction head (L1 regression) with proprioceptive conditioning
- Multi-view image support (primary + wrist cameras)
- Standardized LIBERO benchmark evaluation

We choose OpenVLA-OFT as our validation framework because it provides a strong and reproducible baseline for continuous action prediction, while being modular enough to support the replacement of its action head with our VisionActionHead. The core contributions of this paper—the cloud-edge architecture, the paired-frame training strategy, and the representational specialization analysis—are **framework-agnostic** and applicable to any VLA backbone.

Our modifications to OpenVLA-OFT include:

1. **VisionActionHead module**: A new action head class that combines stale LLM hidden states with real-time SigLIP vision features via a fusion MLP, replacing the original L1RegressionActionHead.

2. **Paired-frame dataset pipeline**: Extension of the RLDS data loading pipeline to extract historical frames from the same episode using the existing window mechanism, enabling delay simulation without cross-episode contamination.

3. **Dual-path forward pass**: Modification of the training loop to run the VLA backbone on both current and delayed frames and compute the dual-path L1 loss.

4. **Delayed evaluation protocol**: Extension of the LIBERO evaluation script to simulate network latency by feeding delayed frames to the VLA while providing current frames to the VisionActionHead.

### 4.2 Benchmark

We evaluate on the **LIBERO** manipulation benchmark, which consists of 4 task suites (Spatial, Object, Goal, 10) of 10 tasks each, with 50 demonstration episodes per task. LIBERO provides a standardized test-bed for evaluating language-conditioned manipulation policies in simulation.

### 4.3 Baselines

We compare against the following baselines:

| Baseline | Description |
|:--|:--|
| **OpenVLA** | The original OpenVLA policy evaluated under the same delayed-observation protocol. |
| **OpenVLA-OFT** | The continuous-action OpenVLA-OFT policy trained without our paired-frame objective. |
| **UniVLA** | A unified VLA baseline evaluated under the same delay-window settings. |
| **CloudEdgeVLA (Ours)** | The VisionActionHead architecture trained with paired fresh/stale cloud features and current edge vision. |

### 4.4 Evaluation Conditions

For closed-loop evaluation, the edge image is always the current observation $o_t$, while the cloud planning feature is delayed by $d \in \{0,5,10,15,20,25,30,35,40\}$ environment steps. The training window is $W=21$, so $d \leq 20$ is within the sampled training support and $d \in \{25,30,35,40\}$ tests extrapolation beyond it. We report aggregate task success on LIBERO-Goal.

For action consistency, we use matched 20k checkpoints of the single-frame baseline and CloudEdgeVLA. Both models receive the same current edge feature $z_t$; only the cloud feature changes from $h_t$ to $h_{t-d}$. The diagnostic covers five LIBERO-Goal tasks, two sampled episodes, and up to 20 samples per task for each delay.

### 4.5 Closed-Loop Delay Robustness

![Closed-loop success and delay-retention summaries.](../results/fig_closed_loop_delay_robustness.png)

**Figure 1: Closed-loop robustness on LIBERO-Goal.** Panel (a) reports task success as cloud-feature delay increases. Panels (b–c) summarize the normalized area under the success-retention curve and the fraction of synchronous success retained at $d=40$. The shaded region $d>20$ lies beyond the delay range sampled during training. Generated by [`plot_closed_loop_delay_robustness.py`](../scripts/plot_closed_loop_delay_robustness.py); exact values are stored in [`fig_closed_loop_delay_robustness_data.json`](../results/fig_closed_loop_delay_robustness_data.json).

| Model | $d=0$ | $d=5$ | $d=10$ | $d=15$ | $d=20$ | $d=25$ | $d=30$ | $d=35$ | $d=40$ |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| OpenVLA | 77.0 | 36.2 | 2.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| OpenVLA-OFT | **97.2** | 76.2 | 26.2 | 15.2 | 4.0 | 2.4 | 0.0 | 0.0 | 0.0 |
| UniVLA | 94.6 | 87.8 | 48.2 | 31.4 | 21.8 | 16.0 | 11.8 | 6.6 | 3.0 |
| **CloudEdgeVLA** | 96.0 | **95.4** | **95.4** | **94.8** | **94.6** | **94.4** | **94.4** | **94.2** | **94.2** |

CloudEdgeVLA preserves synchronous quality while changing the delay-failure profile:

- **Competitive synchronous performance:** at $d=0$, CloudEdgeVLA obtains 96.0%, within 1.2 percentage points of the strongest synchronous result (OpenVLA-OFT at 97.2%).
- **Severe-delay robustness:** at $d=40$, CloudEdgeVLA reaches 94.2%, exceeding the best baseline (UniVLA at 3.0%) by **91.2 percentage points**.
- **Near-flat retention:** success falls by only 1.8 points from $d=0$ to $d=40$, retaining **98.1%** of synchronous success. Its normalized delay AURC is **98.7%**, compared with 36.0% for UniVLA, 22.2% for OpenVLA-OFT, and 12.5% for OpenVLA.
- **Out-of-window generalization:** performance changes from 94.6% at the largest trained delay ($d=20$) to 94.2% at $d=40$. This result supports robustness beyond the sampled training window rather than memorization of the training delay range.

The aggregate success rates currently provide no per-seed uncertainty, so this figure is used as effect-size evidence; confidence intervals should be added when repeated-seed evaluations are available.

### 4.6 Fresh–Stale Action Consistency

To directly test the property induced by the dual-path objective, we measure

$$D(d)=\mathbb{E}\left[\left|g_\phi(h_t,z_t)-g_\phi(h_{t-d},z_t)\right|\right],$$

where actions are compared in normalized action space. Because $z_t$ is fixed, a lower $D(d)$ means that replacing a fresh cloud representation with a stale one causes a smaller action change; lower is better.

![Fresh–stale action consistency for matched checkpoints.](../results/fig_action_consistency_latest.png)

**Figure 2: Fresh–stale action consistency on LIBERO-Goal.** Both curves use matched 20k checkpoints and identical observations. Panel (a) shows mean normalized action drift with one-standard-deviation bands; panel (b) shows the sample distribution at $d=20$. Generated by [`visualize_fresh_stale_action_consistency.py`](../scripts/visualize_fresh_stale_action_consistency.py); summary statistics and per-sample values are stored in [`fig_action_consistency_latest_data.json`](../results/fig_action_consistency_latest_data.json).

| Delay $d$ | Single-frame drift | CloudEdgeVLA drift | Relative reduction |
|--:|--:|--:|--:|
| 1 | 0.0201 | **0.0141** | 30.1% |
| 5 | 0.0577 | **0.0377** | 34.7% |
| 10 | 0.0801 | **0.0548** | 31.6% |
| 15 | 0.0946 | **0.0642** | 32.1% |
| 20 | 0.1048 | **0.0704** | 32.9% |

CloudEdgeVLA has lower drift at every nonzero tested delay. At the maximum trained delay, its mean drift is 0.0704 versus 0.1048 for single-frame training, a **32.9% reduction**. This controlled comparison provides functional evidence for the paper's central mechanism: paired-frame training makes the action output less sensitive to cloud-feature age when real-time edge vision is unchanged. It does not, by itself, identify which hidden dimensions encode semantic or timing information; the claim supported here is output-level compensation rather than internal feature attribution.

Because this diagnostic uses matched 20k checkpoints and a limited trajectory sample, it is treated as mechanistic supporting evidence. The camera-ready analysis should rerun the same protocol on the final matched checkpoints without changing the metric or sampling procedure.

### 4.7 Remaining Ablation Studies

The following architectural and training ablations remain useful complements to the two primary results:

1. **Vision encoder impact:** Compare frozen SigLIP-Base vs. SigLIP-SO400M vs. no edge vision.
2. **Window size:** Vary $W$ and measure robustness beyond each corresponding training-delay support.
3. **Loss weighting:** Compare equal weighting ($\mathcal{L}_{\text{fresh}} + \mathcal{L}_{\text{stale}}$) with weighted variants.
4. **Delay curriculum:** Compare uniform delay sampling with a small-to-large delay curriculum.

The paper grounds its central empirical claims in closed-loop success and the controlled action-consistency intervention above.

### 4.8 Metrics

- **Task Success Rate (%):** Primary closed-loop metric.
- **Normalized delay AURC (%):** Trapezoidal area under $S(d)/S(0)$, normalized by the evaluated delay interval.
- **Success Retention at $d=40$ (%):** $100\,S(40)/S(0)$.
- **Fresh–Stale Action Drift:** Mean absolute difference between actions obtained with fresh and delayed cloud features while holding current edge vision fixed.

---

## 5. Discussion

### 5.1 Paradigm Comparison: Why Our Approach Is Fundamentally Different

Existing dual-system approaches for robot control fall into two categories, both of which impose constraints that our framework avoids:

**Action-chunk re-planning** (e.g., RT-2 + skill policies): The slow system predicts $T$ actions, the fast system executes them, and the slow system replans. This requires the assumption that the $T$-action plan remains valid over its execution horizon—an assumption that is increasingly violated as the slow system's latency grows. With a delay of $k$ steps, the plan is based on information that is $k + T$ steps old by the time the last action is executed.

**Synchronized fast-slow** (e.g., hierarchical IL): The fast system runs at $\alpha \times$ the slow system's frequency and consumes intermediate features from the slow system at each fast tick. This requires: (a) a fixed, predictable $\alpha$ ratio; (b) bounded latency from the slow system; (c) temporal alignment between the two systems' clocks. In networked deployments, none of these conditions are reliably met.

Our approach avoids both failure modes:

| Property | Action-Chunk Replan | Synchronized Fast-Slow | Ours |
|:--|:--|:--|:--|
| Temporal alignment required | Yes (replan timing) | Yes (fixed $\alpha$ ratio) | **No** |
| Delay must be bounded | Yes (plan validity window) | Yes (feature delivery) | **No** |
| Sensitive to network jitter | High (replan timing) | High (missed ticks) | **Low** |
| Edge blocks on cloud | Yes (wait for replan) | Yes (wait for features) | **No** (async) |
| Representation learned | Full action sequence | Intermediate features | **Time-invariant semantics** |

### 5.2 Why Not Reconstruct the Current State?

A natural alternative to our approach is to use a learned dynamics model to predict the current state from delayed observations. However, this requires:

1. An accurate world model (difficult for complex manipulation tasks).
2. Additional inference overhead for the prediction step.
3. Compounding prediction errors for large delays.

Our approach sidesteps these issues by leveraging the edge's direct access to $o_t$ through the local vision encoder. The edge does not need to "predict" the current state—it observes it directly.

### 5.3 Relationship to Ensemble and Multi-View Methods

Our dual-path architecture can be viewed as a form of **late fusion** between two "views" of the environment: the cloud's semantic view (potentially stale) and the edge's perceptual view (always current). This connects to the broader literature on multi-view learning and ensemble methods, where combining diverse information sources improves robustness.

### 5.4 Scaling to Larger Backbones

Our framework is agnostic to the specific VLA backbone. As larger and more capable backbones become available (e.g., 13B, 70B parameters), the cloud-edge split becomes increasingly valuable because:

1. Larger models have higher inference latency, making the async setting more relevant.
2. Larger models produce richer planning features, which the action head can leverage more effectively.
3. The edge-side components (vision encoder + action head) remain lightweight regardless of backbone size.

### 5.5 Limitations

1. **Double backbone compute during training**: Our paired-frame strategy requires two backbone forward passes per step. This can be mitigated by batching.
2. **Vision encoder capacity**: The frozen SigLIP encoder may not capture all task-relevant visual information (e.g., fine-grained depth or force cues). Future work could explore task-adapted vision encoders.
3. **Simulation only**: Current experiments are in simulation. Real-world deployment would introduce additional challenges (sensor noise, sim-to-real gap, variable network conditions).

---

## 6. Conclusion

We present a latency-tolerant cloud-edge collaborative VLA framework that addresses a fundamental challenge in asynchronous robot deployment: how to maintain real-time control responsiveness when the high-level planning backbone operates under network latency. Our key insight is that this challenge is best addressed not through temporal alignment mechanisms, but through **representational specialization**—training the cloud backbone to produce time-invariant semantic features and the edge action head to ground these features into precise motor commands using real-time vision. The paired-frame dual-path training strategy achieves this specialization naturally, without requiring explicit architectural constraints or delay estimation. Compared to existing dual-system approaches that rely on action-chunk re-planning or synchronized fast-slow pipelines, our framework is simpler, more robust to network variability, and scales gracefully to larger backbone architectures.

---

## Appendix A: Notation Reference

| Symbol | Description |
|:--|:--|
| $o_t$ | Observation at environment step $t$ |
| $\ell$ | Language instruction |
| $a_t$ | Ground-truth action at step $t$ |
| $\hat{a}_t$ | Predicted action at step $t$ |
| $f_\theta$ | Cloud-side VLA backbone (parameters $\theta$) |
| $g_\phi$ | Edge-side action head (parameters $\phi$) |
| $v_\psi$ | Edge-side vision encoder (parameters $\psi$, frozen) |
| $h$ | Planning features (hidden states from backbone) |
| $z_t$ | Real-time vision features from $v_\psi(o_t)$ |
| $W$ | Training window size (number of consecutive frames per sample) |
| $d$ | Simulated delay (number of environment steps) |
| $T$ | Action chunk length |
| $A$ | Action dimensionality |
| $D$ | Backbone hidden dimension |
| $D_v$ | Vision encoder feature dimension |
| $I_{\text{inv}}$ | Delay-invariant mutual information |
| $I_{\text{delay}}$ | Delay-sensitive mutual information |

## Appendix B: Hyperparameter Guide

| Parameter | Recommended Range | Description |
|:--|:--|:--|
| $W$ (window_size) | $d_{\max} + 1$ | Must be at least max_delay + 1 |
| Learning rate | $5 \times 10^{-4}$ | With LoRA fine-tuning |
| LoRA rank | 16–32 | Trade-off between capacity and efficiency |
| Vision encoder | SigLIP-Base | Frozen; SigLIP-SO400M for higher capacity |
| Batch size | 8 per GPU | Scale with gradient accumulation |
| Max training steps | 100k–200k | Depends on dataset size |

## Appendix C: Deployment Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                      CLOUD SERVER                         │
│                  System 2: Latency-Insensitive             │
│                                                           │
│   o_{t-k} ──→ VLA Backbone f_θ ──→ h_{t-k}              │
│              (7B params, LoRA)    (planning features)     │
│              Encodes: task goals, strategy, semantics     │
│              (time-invariant information)                  │
│                        │                                  │
│                        │ network (async, variable latency)│
└────────────────────────┼─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│                      EDGE ROBOT                           │
│                  System 1: Latency-Sensitive               │
│                                                           │
│   o_t ──→ Vision Encoder v_ψ ──→ z_t                    │
│          (SigLIP-Base, frozen)   (vision features)       │
│          Encodes: spatial details, contact state,         │
│          perturbations (time-critical information)        │
│                                                           │
│   h_{t-k} + z_t ──→ Action Head g_φ ──→ a_t             │
│                    (lightweight MLP)   (action chunk)     │
│                    Grounds high-level plan into           │
│                    real-time motor commands                │
│                          │                                │
│                          ▼                                │
│                     Robot Controller                      │
└──────────────────────────────────────────────────────────┘
```
