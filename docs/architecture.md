# Architecture Design Document

> Reference guide for drawing the system architecture figure.
> Last updated: 2026-07-05

---

## 1. Overview: Three Diagrams to Draw

We recommend producing **three diagrams** for the paper:

| Diagram | Purpose | Content |
|:--|:--|:--|
| **Fig. 1 — System Architecture** | High-level overview of cloud-edge components and deployment flow | Components, data flow, latency annotation |
| **Fig. 2 — Training Pipeline** | Paired-frame dual-path training with gradient flow | Two forward passes, loss computation, what gets updated |
| **Fig. 3 — Representational Specialization** | What each component learns | Feature space visualization or information-theoretic diagram |

Below we describe each diagram in exhaustive detail.

---

## 2. Fig. 1 — System Architecture (Deployment)

### 2.1 Purpose

Show the reader the complete cloud-edge system at a glance: what runs where, what data flows in which direction, and where latency occurs.

### 2.2 Layout

Use a **vertical layout** with two major regions separated by a horizontal dashed line representing the network boundary:

```
┌──────────────────────────────────────┐
│           CLOUD SERVER               │  ← top region (gray/blue background)
│                                      │
│                                      │
└──────────────────────────────────────┘
- - - - - - - network boundary - - - - -  ← dashed horizontal line, annotate "Network Latency Δt"
┌──────────────────────────────────────┐
│           EDGE ROBOT                 │  ← bottom region (green/white background)
│                                      │
│                                      │
└──────────────────────────────────────┘
```

### 2.3 Components (left to right within each region)

#### Cloud Region (top)

**Box A — VLA Backbone $f_\theta$**
- Large box, visually dominant (to convey computational weight)
- Label inside: "VLA Backbone $f_\theta$" with subtitle "7B params, LoRA"
- Icon: a transformer/LM icon or a brain icon
- Color: deep blue

**Sub-components within Box A** (shown as internal layers, left to right):

```
┌─────────────────────────────────────────────────────────┐
│                  VLA Backbone f_θ                        │
│                                                         │
│  ┌──────────┐    ┌──────────┐    ┌───────────────────┐  │
│  │ Vision    │    │ Language  │    │ Transformer LLM   │  │
│  │ Encoder   │───→│ Embedding │───→│ (LoRA-adapted)    │  │
│  │ (SigLIP + │    │ + Prompt  │    │                   │  │
│  │  DINOv2)  │    │           │    │ → hidden states h │  │
│  └──────────┘    └──────────┘    └───────────────────┘  │
│       ↑                                                │
│   o_{t-k}                                               │
│   (delayed                                              │
│    observation)                                         │
└─────────────────────────────────────────────────────────┘
```

- **Vision Encoder**: SigLIP + DINOv2 dual-encoder (existing in OpenVLA). Input: delayed observation image $o_{t-k}$.
- **Language Embedding**: Tokenizes instruction $\ell$, builds prompt.
- **Transformer LLM**: The core backbone. Produces hidden states. Highlight the **action-token positions** in the output sequence (these become $h$).
- Output arrow: $h_{t-k} \in \mathbb{R}^{L \times D}$, label "planning features (stale)"

**Box B — Async Queue**
- Small box between cloud and network boundary
- Label: "Async Output Buffer"
- Purpose: shows that the cloud sends features asynchronously, the latest available features are used
- Icon: a FIFO queue symbol

#### Network Boundary

- Dashed line across the full width
- Annotate with: "Network Latency: $\Delta t$ (variable)"
- Two arrows crossing the boundary:
  - **Upward arrow** (edge → cloud): "observation $o_t$" (send to cloud for processing)
  - **Downward arrow** (cloud → edge): "planning features $h_{t-k}$" (may be stale)
- Use a "clock skew" symbol or a small delay icon to emphasize staleness

#### Edge Region (bottom)

Three boxes from left to right:

**Box C — Vision Encoder $v_\psi$**
- Medium box
- Label: "Edge Vision Encoder $v_\psi$" with subtitle "SigLIP-Base, frozen"
- Input arrow from observation icon: "$o_t$ (current, real-time)"
- Output arrow: "$z_t$ (real-time vision features)"
- Color: green (to contrast with cloud's blue)
- Mark with a "⚡ real-time" badge

**Box D — Action Head $g_\phi$**
- Medium box, same visual weight as Box C
- Label: "VisionActionHead $g_\phi$" with subtitle "Fusion MLP"
- Two input arrows:
  - From cloud: "$h_{t-k}$ (stale planning features)" — use a faded/dashed arrow to convey staleness
  - From vision encoder: "$z_t$ (real-time vision)" — use a solid/bold arrow to convey freshness
- Internal detail (optional, for a zoom-in callout):
  ```
  ┌─────────────────────────────────────┐
  │         VisionActionHead g_φ        │
  │                                     │
  │  h_{t-k}        z_t                │
  │     │              │                │
  │     ▼              ▼                │
  │  MeanPool      Projector            │
  │  (L×D→T×D)     (Dv→D)              │
  │     │              │                │
  │     └──── Cat ─────┘                │
  │          (T×2D)                     │
  │            │                        │
  │       Fusion MLPResNet              │
  │            │                        │
  │            ▼                        │
  │        â_t (T×A)                    │
  └─────────────────────────────────────┘
  ```
- Output arrow: "$\hat{a}_t$ (action chunk)"
- Color: orange or amber

**Box E — Robot Controller**
- Small box at the far right
- Label: "Robot Controller"
- Input: action chunk $\hat{a}_t$
- Output: downward arrow to "Environment" with label "execute action"
- Icon: a robot arm or gripper

**Observation Source**
- Icon at the far left of edge region: camera/sensor icon
- Label: "Environment $o_t$"
- Two outgoing arrows:
  - One to cloud (upward, through network): "send to cloud"
  - One to edge vision encoder (rightward): "real-time"

### 2.4 Annotations and Callouts

Add the following text callouts around the diagram:

1. **On the cloud side**: "System 2: Latency-insensitive. Encodes task goals, strategy, semantics. Output may be delayed by $\Delta t$."

2. **On the edge side**: "System 1: Latency-sensitive. Real-time visual grounding. Never blocks on cloud."

3. **On the action head**: "Fuses stale planning ($h_{t-k}$) with real-time vision ($z_t$). Learns to compensate for delay."

4. **On the network boundary**: "Asynchronous, non-blocking. Edge always uses the most recently received features."

### 2.5 Color Scheme Suggestion

| Element | Color | Meaning |
|:--|:--|:--|
| Cloud components | Blue (#4A90D9) | Compute-heavy, remote |
| Edge components | Green (#5CB85C) | Lightweight, local |
| Action head | Amber (#F0AD4E) | Fusion point, the key contribution |
| Stale data flow | Gray dashed | Delayed, potentially outdated |
| Fresh data flow | Green solid | Real-time, always current |
| Network boundary | Red dashed | Source of the problem |

---

## 3. Fig. 2 — Training Pipeline (Paired-Frame Dual-Path)

### 3.1 Purpose

Show the complete training forward pass: how paired frames are extracted, processed through the backbone, fused with vision, and how the dual-path loss is computed. This is the most detailed diagram.

### 3.2 Layout

Use a **horizontal flow** from left to right, with two parallel paths (fresh and stale) that converge at the loss computation.

```
Left: Data → Middle: Backbone → Right: Action Head → Far Right: Loss
```

### 3.3 Detailed Flow

#### Stage 1: Paired Frame Extraction (leftmost)

```
┌─────────────────────────────────────┐
│     Dataset Window (same episode)   │
│                                     │
│  o_{t-W+1}  o_{t-W+2}  ...  o_t    │
│     past                        now │
│                                     │
│  ┌─────────────────┐               │
│  │ Sample d~U(1,W-1)│              │
│  └────────┬────────┘               │
│           │                         │
│     ┌─────┴──────┐                 │
│     ▼            ▼                  │
│  o_{t-d}        o_t                │
│  (delayed)     (current)           │
└─────────────────────────────────────┘
```

- Draw a horizontal bar representing the window of $W$ frames
- Mark the current frame $o_t$ at the right end (highlighted)
- Mark a random past frame $o_{t-d}$ somewhere in the middle (selected by random arrow)
- Label: "Same episode, guaranteed by dataset window mechanism"
- Split into two output arrows: "delayed frame" (gray) and "current frame" (green)

#### Stage 2: Dual VLA Forward Pass (middle-left)

Two parallel boxes, one for each path:

```
      o_{t-d}                          o_t
        │                               │
        ▼                               ▼
  ┌───────────┐                   ┌───────────┐
  │ VLA f_θ   │                   │ VLA f_θ   │
  │ (shared   │                   │ (shared   │
  │  weights) │                   │  weights) │
  └─────┬─────┘                   └─────┬─────┘
        │                               │
        ▼                               ▼
     h_stale                          h_fresh
  (stale planning)               (fresh planning)
```

- **Critical**: both passes share the **same** backbone $f_\theta$ (same weights). Draw them as two instances of the same box, or one box with two input/output arrows.
- Annotate: "Shared backbone, two forward passes"
- The stale path: use a gray/faded arrow
- The fresh path: use a green solid arrow

#### Stage 3: Edge Vision (middle, single path)

```
        o_t
        │
        ▼
  ┌───────────┐
  │ v_ψ       │
  │ (SigLIP,  │
  │  frozen)  │
  └─────┬─────┘
        │
        ▼
       z_t
  (real-time vision)
```

- Single box, drawn **between** the two backbone paths (to show it feeds into both action heads)
- Arrow splits into two: one going to the fresh action head, one to the stale action head
- Label: "Frozen, real-time, shared between both paths"

#### Stage 4: Dual Action Head (middle-right)

Two parallel action head instances:

```
  h_stale    z_t              h_fresh    z_t
     │        │                  │        │
     └──┬─────┘                  └──┬─────┘
        │                           │
        ▼                           ▼
  ┌───────────┐               ┌───────────┐
  │ g_φ       │               │ g_φ       │
  │ (shared   │               │ (shared   │
  │  weights) │               │  weights) │
  └─────┬─────┘               └─────┬─────┘
        │                           │
        ▼                           ▼
    â_stale                       â_fresh
```

- Again, shared weights — two instances of the same module
- The stale path's $h_{t-k}$ arrow should be visually thinner/faded
- The fresh path's $h_t$ arrow should be solid
- $z_t$ arrows to both should be solid green (always real-time)

#### Stage 5: Dual-Path Loss (rightmost)

```
  â_stale   a_gt              â_fresh   a_gt
     │        │                  │        │
     ▼        ▼                  ▼        ▼
  ┌──────────────┐           ┌──────────────┐
  │ L_stale =    │           │ L_fresh =    │
  │ ‖â_s - a_gt‖₁│           │ ‖â_f - a_gt‖₁│
  └──────┬───────┘           └──────┬───────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
              ┌──────────┐
              │ L_total  │
              │ = L_s+L_f│
              └────┬─────┘
                   │
                   ▼
              Backprop to θ, φ
              (ψ frozen)
```

- Ground truth $a_t$ appears twice (once per path), drawn as a shared reference
- The two losses merge at a summation node
- Gradient arrows go back to $\theta$ (backbone) and $\phi$ (action head)
- Explicitly mark: "$\psi$ frozen — no gradient"

#### Stage 6: Gradient Flow Annotations

Draw gradient arrows with labels:

- From $\mathcal{L}_{\text{fresh}}$:
  - → $\theta$: "Direct action supervision to backbone"
  - → $\phi$: "Trains action head on fresh features"
- From $\mathcal{L}_{\text{stale}}$:
  - → $\theta$: "Encourages time-invariant representations" (through $h^{\text{stale}}$)
  - → $\phi$: "Trains action head to compensate for stale features using $z_t$"

### 3.4 Color Coding

| Path | Color | Meaning |
|:--|:--|:--|
| Fresh path (current frame) | Green solid | Always real-time, primary path |
| Stale path (delayed frame) | Gray/faded | Simulated delay, secondary path |
| Vision features $z_t$ | Green solid | Always real-time, shared |
| Ground truth $a_t$ | Black | Reference, unchanged |
| Gradient flow | Red arrows | Backward pass |

---

## 4. Fig. 3 — Representational Specialization (Information-Theoretic View)

### 4.1 Purpose

Illustrate the key insight: the dual-path training induces a division of labor between the backbone (time-invariant semantics) and the action head (time-critical visual-motor control).

### 4.2 Layout

Use **two side-by-side feature space diagrams** (before vs. after our training), or a single diagram showing the information decomposition.

#### Option A: Two-Column Comparison

**Left column — Standard VLA Training (fragile under delay)**

```
┌─────────────────────────────┐
│   VLA Backbone f_θ          │
│   Encodes EVERYTHING:       │
│   ┌───────────────────────┐ │
│   │ ● Task semantics      │ │
│   │ ● Object positions    │ │
│   │ ● Gripper state       │ │
│   │ ● Contact forces      │ │
│   │ ● Timing information  │ │  ← ALL mixed together
│   └───────────────────────┘ │
│              │               │
│              ▼               │
│   Action Head: simple MLP   │
│   (no visual grounding)     │
│              │               │
│              ▼               │
│         Actions              │
└─────────────────────────────┘
→ If input is delayed: EVERYTHING is stale → failure
```

**Right column — Our Training (robust under delay)**

```
┌──────────────────────┐    ┌──────────────────────┐
│  VLA Backbone f_θ    │    │  VisionActionHead g_φ │
│  Encodes ONLY:       │    │  Encodes:             │
│  ┌────────────────┐  │    │  ┌──────────────────┐ │
│  │ Task semantics │  │    │  │ Spatial details   │ │
│  │ Goal direction │  │    │  │ Gripper state     │ │
│  │ Strategy       │  │    │  │ Contact state     │ │
│  │ (time-invariant│  │    │  │ Perturbations     │ │
│  │  information)  │  │    │  │ (time-critical    │ │
│  └────────────────┘  │    │  │  information)     │ │
│         │            │    │  └──────────────────┘ │
│         │  h (stale OK│    │         ↑             │
│         └────────────┼───→│    z_t (real-time)    │
│                      │    │         │             │
└──────────────────────┘    │         ▼             │
                            │      Actions          │
                            └──────────────────────┘
→ If h is delayed: only stale semantics → vision compensates
```

#### Option B: Venn Diagram / Information Decomposition

Draw a Venn diagram showing the information in the action prediction:

```
    ┌─────────────────────────────────────────────┐
    │           Total Information for a_t          │
    │                                              │
    │   ┌──────────────────┐                       │
    │   │ I_inv (h; a_t)   │                       │
    │   │ Delay-invariant  │                       │
    │   │ from backbone    │                       │
    │   │ (task semantics) │                       │
    │   └────────┬─────────┘                       │
    │            │ overlap                          │
    │   ┌────────┴─────────┐                       │
    │   │ I(z_t; a_t|h)   │                       │
    │   │ From vision      │                       │
    │   │ (time-critical)  │                       │
    │   └──────────────────┘                       │
    │                                              │
    │   Training pushes backbone to encode I_inv   │
    │   and action head to extract the rest from z │
    └─────────────────────────────────────────────┘
```

---

## 5. Supplementary Diagram: Async Deployment Timeline

### 5.1 Purpose

Show how the system operates over time during deployment, illustrating the asynchronous nature of cloud-edge interaction.

### 5.2 Layout

A **sequence diagram** with time flowing downward:

```
Time    Cloud                    Network              Edge (Robot)
────    ─────                    ───────              ────────────
t=0     Receive o_0 ──────────→  Process
        Start inference
        (takes Δt_cloud)

t=1                                  ← o_1 sent        Capture o_1
                                                       Compute z_1
                                                       (no h yet,
                                                        use default
                                                        or wait)

t=2                                  ← o_2 sent        Capture o_2
                                                       Compute z_2

t=3     h_0 ready ──────────────→                     Receive h_0
        Send h_0                                      a_2 = g(h_0, z_2)
                                                      Execute a_2
                                                      (h_0 is 3 steps
                                                       stale, but z_2
                                                       is real-time)

t=4                                  ← o_4 sent        Capture o_4
                                                       Compute z_4

t=5     h_2 ready ──────────────→                     Receive h_2
        Send h_2                                      a_4 = g(h_2, z_4)
                                                      Execute a_4
```

Key observations to annotate:
1. Edge **never blocks** — it always computes $z_t$ and executes actions at every step
2. Cloud latency $\Delta t_{\text{cloud}}$ varies (3 steps for first inference, 3 steps for second)
3. The action head compensates: when $h$ is 3 steps stale, $z_t$ provides the missing information
4. The edge uses the **most recently received** $h$, which may be arbitrarily stale

---

## 6. Supplementary Diagram: Comparison with Existing Approaches

### 6.1 Purpose

Visually contrast our approach with the two existing paradigms.

### 6.2 Layout

Three parallel columns:

```
  Action-Chunk Replan       Sync Fast-Slow           Ours
  ──────────────────       ──────────────           ────

  Cloud: plan T steps       Cloud: features          Cloud: features
       │                         │                        │
       ▼                         ▼                        ▼
  Execute T steps           Fast sys ×α               Edge: fuse with
  (open-loop)               (strict ratio)            real-time vision
       │                         │                        │
       ▼                         ▼                        ▼
  Wait for replan           Wait for next tick        Execute (never waits)
       │                         │                        │
  [STALL if delayed]        [STALL if delayed]        [NEVER STALLS]

  Assumes:                  Assumes:                  Assumes:
  - Plan valid for T steps  - Fixed α ratio           - Nothing!
  - Bounded latency         - Bounded latency         - Works with any
  - Sync clock              - Sync clock                delay ≤ training max
```

---

## 7. Drawing Guidelines

### 7.1 General Style

- Use **rounded rectangles** for all components
- Use **arrows** for data flow (solid = real-time, dashed = potentially stale)
- Use **different line widths** to convey importance (thick for fresh, thin for stale)
- Use **color coding** consistently across all figures
- Include **mathematical notation** ($h_{t-k}$, $z_t$, etc.) on arrows and inside boxes

### 7.2 Suggested Tools

| Tool | Best for | Notes |
|:--|:--|:--|
| TikZ (LaTeX) | Publication-quality vector figures | Best for Fig. 1 and Fig. 2 |
| draw.io / diagrams.net | Quick prototyping | Good for iterating on layout |
| Figma / Sketch | Polished design | Good if you want a modern aesthetic |
| matplotlib + patches | Programmatic generation | Good for Fig. 3 (feature space) |
| OmniGraffle | Mac users | Excellent for sequence diagrams (Fig. 5) |

### 7.3 Font and Size Recommendations

- Component labels: 10–12pt bold
- Arrow labels / math: 9–10pt italic
- Annotations / callouts: 8–9pt regular
- Figure width: ≤ 17cm (single column) or ≤ 8.5cm (double column for AAAI)
- Ensure all text is legible when printed in grayscale (test by converting to grayscale)

### 7.4 Figure Caption Templates

**Fig. 1**: "System architecture of our latency-tolerant cloud-edge collaborative VLA framework. The cloud-side VLA backbone $f_\theta$ produces high-level planning features $h_{t-k}$ from (potentially delayed) observations. The edge-side VisionActionHead $g_\phi$ fuses these features with real-time local vision features $z_t$ from the frozen encoder $v_\psi$ to produce latency-robust actions $\hat{a}_t$. The cloud operates asynchronously; the edge never blocks."

**Fig. 2**: "Paired-frame dual-path training pipeline. Each training step samples a delayed frame $o_{t-d}$ and the current frame $o_t$ from the same episode window. Both are processed through the shared backbone $f_\theta$, and both resulting planning features are fused with the shared real-time vision features $z_t$. The dual-path L1 loss $\mathcal{L} = \mathcal{L}_{\text{fresh}} + \mathcal{L}_{\text{stale}}$ trains the backbone to produce delay-invariant representations and the action head to compensate for stale features using vision."

**Fig. 3**: "Emergent representational specialization. Standard VLA training (left) encodes all task-relevant information—including timing-critical details—into the backbone's hidden states, creating a fragile dependency on observation freshness. Our paired-frame training (right) encourages the backbone to retain only time-invariant semantic information (task goals, strategy), while the action head learns to extract time-critical details (spatial precision, contact state) from real-time vision features $z_t$."

---

## 8. Checklist Before Submission

- [ ] Fig. 1 clearly shows cloud vs. edge boundary
- [ ] Fig. 1 shows both data directions (obs up, features down)
- [ ] Fig. 1 annotates network latency
- [ ] Fig. 2 shows both fresh and stale paths with distinct visual styles
- [ ] Fig. 2 shows that backbone and action head weights are shared
- [ ] Fig. 2 shows gradient flow with distinct annotations for each loss term
- [ ] Fig. 2 marks $\psi$ as frozen (no gradient)
- [ ] Fig. 3 contrasts standard training vs. our approach
- [ ] All figures are legible in grayscale
- [ ] All figures use consistent notation ($h$, $z_t$, $o_t$, $a_t$, etc.)
- [ ] Figure widths conform to AAAI formatting guidelines
