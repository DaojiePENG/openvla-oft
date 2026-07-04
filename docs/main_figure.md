# Paper Main Figure Design (Fig. 1)

> This is the single comprehensive figure for the paper. It should fit in one full-width column (17cm × ~14cm) and tell the complete story at a glance.

---

## Overall Layout

The figure has **three horizontal panels** stacked vertically, connected by visual flow:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│   PANEL A: Deployment Architecture (top, ~35% height)                           │
│   "How the system runs at test time"                                            │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   PANEL B: Paired-Frame Training Pipeline (middle, ~45% height)                 │
│   "How the system learns"                                                       │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   PANEL C: Emergent Representational Specialization (bottom, ~20% height)       │
│   "What each component learns"                                                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────┐
│  Panel A (top): 部署架构                 │  ← "系统怎么跑"
│  Cloud VLA ──延迟──→ Edge ActionHead     │
│  stale h (灰虚线) + fresh z_t (绿实线)    │
├─────────────────────────────────────────┤
│  Panel B (middle): 训练流程              │  ← "系统怎么学"
│  同episode窗口 → 采样延迟帧              │
│  双路VLA → 双路ActionHead → 双路Loss     │
│  梯度回传标注：谁学什么                   │
├─────────────────────────────────────────┤
│  Panel C (bottom): 表征分离              │  ← "学到了什么"
│  标准VLA: 全部混一起 → 脆弱              │
│  我们: 骨干=语义, Head=时序 → 鲁棒       │
└─────────────────────────────────────────┘
```
---

## Panel A: Deployment Architecture (Top)

### Spatial Layout (left to right)

```
    Environment                                              Environment
    (o_t)                                                        │
       │                                                         │
       ├──────────────── send to cloud ──────────────┐           │
       │                                             │           │
       │            ┌────────────────────────────────┼───────┐   │
       │            │  CLOUD SERVER                 │       │   │
       │            │                                │       │   │
       │            │   ┌────────────────────────┐   │       │   │
       │            │   │    VLA Backbone f_θ     │   │       │   │
       │            │   │    (7B, LoRA)           │   │       │   │
       │            │   │                         │   │       │   │
       │            │   │  o_{t-k} → [SigLIP +   │   │       │   │
       │            │   │   DINOv2] → [LLM] → h  │   │       │   │
       │            │   └────────────┬────────────┘   │       │   │
       │            │                │                 │       │   │
       │            │        h_{t-k} (planning        │       │   │
       │            │         features)                │       │   │
       │            └────────────────┼─────────────────┘       │   │
       │                             │                         │   │
       │                    ═════════╪═════════                │   │
       │                    Network Latency Δt                 │   │
       │                    ═════════╪═════════                │   │
       │                             │                         │   │
       │            ┌────────────────┼─────────────────────┐   │   │
       │            │  EDGE ROBOT    │                     │   │   │
       │            │                ▼                     │   │   │
       │   ┌────────┼──┐  ┌──────────────────┐            │   │   │
       ├───┤ v_ψ    │  │  │  Action Head g_φ  │            │   │   │
       │   │(SigLIP │  │  │                  │            │   │   │
       │   │ frozen)│  │  │  h_{t-k} + z_t   │──→ a_t ────┼───┘   │
       │   │   │    │  │  │       ↓          │            │       │
       │   │   z_t  │──┼─→│  Fusion MLP      │            │       │
       │   └────────┼──┘  └──────────────────┘            │       │
       │            │                                      │       │
       │            └──────────────────────────────────────┘       │
       │                                                           │
```

### Visual Design Instructions

**Cloud region:**
- Background: light blue rectangle, rounded corners
- The VLA Backbone box is large and prominent (to convey computational weight)
- Inside the backbone, show a simplified pipeline: Image → Vision Encoder → LLM → hidden states
- The backbone box has a subtle "7B" badge

**Network boundary:**
- A horizontal dashed red line spanning the full width
- Label above the line: "Network Latency $\Delta t$"
- The downward arrow ($h_{t-k}$) crosses this line — make it a **dashed gray arrow** to convey staleness
- The upward arrow ($o_t$) also crosses — make it a **thin solid arrow**

**Edge region:**
- Background: light green rectangle
- Two main components side by side:
  - **Vision Encoder $v_\psi$** (left): smaller box, green border, "frozen" badge, "⚡ real-time" label
  - **Action Head $g_\phi$** (right): medium box, amber/orange border, "Fusion MLP" label

**Key arrows:**
- $o_t$ → cloud: thin, going up-right
- $h_{t-k}$ ← cloud: **thick dashed gray**, coming down into action head (label: "$h_{t-k}$ stale planning")
- $o_t$ → $v_\psi$ → $z_t$ → action head: **thick solid green** (label: "$z_t$ real-time vision")
- Action head → robot arm: solid black arrow, label "$\hat{a}_t$"

**Annotations (text callouts):**
- Above cloud box: "System 2: Latency-insensitive" in blue text
- Above edge box: "System 1: Latency-sensitive" in green text
- A small "stale" label on the $h_{t-k}$ arrow with a clock icon
- A small "real-time" label on the $z_t$ arrow with a lightning icon

---

## Panel B: Paired-Frame Training Pipeline (Middle)

### Spatial Layout (left to right, with two parallel paths)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│   Dataset Window (same episode)                                                 │
│   ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐                                              │
│   │  │  │  │▓▓│  │  │  │  │  │██│  ← ██ = current frame o_t                    │
│   └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘    ▓▓ = sampled delayed frame o_{t-d}        │
│                  ↑            ↑                                                  │
│                  │            │                                                  │
│             o_{t-d}         o_t                                                 │
│               │              │                                                  │
│      ┌────────┴──┐    ┌─────┴─────┐                                            │
│      │  Stale    │    │  Fresh     │                                            │
│      │  Path     │    │  Path      │                                            │
│      │           │    │            │                                            │
│      ▼           │    ▼            │                                            │
│  ┌────────┐      │  ┌────────┐     │                                            │
│  │ VLA f_θ│      │  │ VLA f_θ│     │   ← SAME weights θ                       │
│  └───┬────┘      │  └───┬────┘     │                                            │
│      │           │      │          │                                            │
│      ▼           │      ▼          │                                            │
│   h_{stale}      │   h_{fresh}     │                                            │
│      │           │      │          │                                            │
│      │    ┌──────┘      │          │                                            │
│      │    │  z_t ← v_ψ(o_t)       │   ← frozen, shared                        │
│      │    │    │         │          │                                            │
│      ▼    ▼    ▼         ▼    ▼     │                                            │
│  ┌──────────┐      ┌──────────┐     │                                            │
│  │ Action   │      │ Action   │     │                                            │
│  │ Head g_φ │      │ Head g_φ │     │   ← SAME weights φ                       │
│  └────┬─────┘      └────┬─────┘     │                                            │
│       │                 │           │                                            │
│       ▼                 ▼           │                                            │
│    â_stale           â_fresh        │                                            │
│       │                 │           │                                            │
│       ▼                 ▼           │                                            │
│  ┌─────────┐       ┌─────────┐      │                                            │
│  │L_stale = │       │L_fresh = │      │                                            │
│  │‖â_s-a_gt‖│       │‖â_f-a_gt‖│      │                                            │
│  └────┬─────┘       └────┬─────┘      │                                            │
│       │                  │           │                                            │
│       └──────── + ────────┘           │                                            │
│               │                      │                                            │
│               ▼                      │                                            │
│         L = L_s + L_f                │                                            │
│               │                      │                                            │
│               ▼                      │                                            │
│       Backprop to θ, φ               │                                            │
│       (ψ frozen ✗)                   │                                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Visual Design Instructions

**Top bar — Dataset Window:**
- A horizontal bar of small squares representing consecutive frames
- The rightmost square is **filled green** (current frame $o_t$)
- A randomly selected middle square is **filled orange** (delayed frame $o_{t-d}$)
- An arrow with label "$d \sim U(1, W-1)$" connects from a "random" icon to the orange square
- Label: "Same episode window" (emphasize: NOT cross-episode)

**Two parallel paths:**
- **Stale path** (top): arrows and boxes in **gray/light color** to convey "delayed"
  - $o_{t-d}$ → VLA → $h^{\text{stale}}$ → Action Head → $\hat{a}^{\text{stale}}$
- **Fresh path** (bottom): arrows and boxes in **green/bright color** to convey "real-time"
  - $o_t$ → VLA → $h^{\text{fresh}}$ → Action Head → $\hat{a}^{\text{fresh}}$
- Both VLA boxes should be visually the **same box** (draw one box, split the arrow into it) with label "shared $\theta$"
- Both Action Head boxes should be visually the **same box** with label "shared $\phi$"

**Vision encoder:**
- A single small green box labeled "$v_\psi$ (frozen)"
- Input: $o_t$ (from the current frame only)
- Output: $z_t$, which fans out to **both** action heads
- Draw the $z_t$ arrows as **solid green**, connecting to both paths

**Ground truth:**
- $a_t$ (ground truth action) appears as a **black reference box** on the right
- Arrows from $\hat{a}^{\text{stale}}$ and $\hat{a}^{\text{fresh}}$ both point to $a_t$
- L1 loss symbols at the junction

**Loss and gradient:**
- The summation $\mathcal{L} = \mathcal{L}_{\text{fresh}} + \mathcal{L}_{\text{stale}}$ at the bottom
- Gradient arrows going **up** to $\theta$ and $\phi$
- A **crossed-out** arrow to $\psi$ (frozen, no gradient)

**Gradient annotations** (small text labels on gradient arrows):
- $\nabla_\theta \mathcal{L}_{\text{stale}}$: "Learn time-invariant representations"
- $\nabla_\phi \mathcal{L}_{\text{stale}}$: "Compensate for stale $h$ using $z_t$"
- $\nabla_\theta \mathcal{L}_{\text{fresh}}$: "Direct action supervision"
- $\nabla_\phi \mathcal{L}_{\text{fresh}}$: "Standard action mapping"

---

## Panel C: Emergent Representational Specialization (Bottom)

### Layout

A horizontal comparison with a divider in the middle:

```
┌─────────────────────────────┬──────────────────────────────────┐
│    Before: Standard VLA     │    After: Our Paired-Frame Train  │
│                             │                                   │
│   ┌───────────────────┐     │   ┌──────────┐   ┌────────────┐  │
│   │ VLA Backbone      │     │   │ Backbone  │   │ ActionHead │  │
│   │                   │     │   │           │   │            │  │
│   │ ● Task goals      │     │   │ ● Goals   │   │ ● Spatial  │  │
│   │ ● Object pos      │     │   │ ● Strategy│   │ ● Contact  │  │
│   │ ● Gripper state   │     │   │ ● Semantics│  │ ● Gripper  │  │
│   │ ● Contact forces  │     │   │           │   │ ● Timing   │  │
│   │ ● Timing info     │     │   │ (invariant)│  │ (critical) │  │
│   │                   │     │   └──────────┘   └────────────┘  │
│   │ ALL mixed together│     │          ↑             ↑         │
│   └───────────────────┘     │          h           z_t         │
│                             │      (stale OK)   (real-time)    │
│   → Fragile under delay     │   → Robust under delay           │
└─────────────────────────────┴──────────────────────────────────┘
```

### Visual Design Instructions

**Left column — Standard VLA:**
- A single large box containing **all** information types (mixed colors)
- Use different colored dots/icons for each info type
- Label: "Standard Training"
- Below: red text "⚠ Fragile: all info in stale $h$ → fails under delay"

**Right column — Our approach:**
- Two boxes side by side:
  - **Backbone box**: only contains high-level items (goals, strategy, semantics) in blue
  - **ActionHead box**: contains low-level items (spatial, contact, gripper, timing) in green
- Arrows from backbone ($h$) and vision ($z_t$) feed into the action head
- Below: green text "✓ Robust: $h$ carries delay-invariant info, $z_t$ fills in the rest"

**Divider:**
- A large arrow from left to right, labeled "Our Paired-Frame Training"
- Or: a "$\Rightarrow$" symbol with the label

---

## Color Palette (Consistent Across All Panels)

| Element | Color | Hex | Usage |
|:--|:--|:--|:--|
| Cloud components | Steel Blue | `#4682B4` | VLA backbone, cloud background |
| Edge components | Sea Green | `#2E8B57` | Vision encoder, edge background |
| Action Head | Dark Orange | `#E8830A` | Fusion module (key contribution) |
| Stale / delayed data | Light Gray | `#B0B0B0` | Dashed arrows, stale path |
| Fresh / real-time data | Forest Green | `#228B22` | Solid arrows, fresh path |
| Ground truth | Black | `#000000` | $a_t$, loss |
| Network boundary | Red | `#CC0000` | Dashed line, latency label |
| Gradient arrows | Crimson | `#DC143C` | Backward pass |
| Frozen (no gradient) | Gray + ✗ | `#808080` | Crossed-out arrow |

---

## Sizing Guide (for AAAI 17cm full-width figure)

| Panel | Height | Width |
|:--|:--|:--|
| Panel A: Deployment | ~5cm | 17cm |
| Panel B: Training | ~6.5cm | 17cm |
| Panel C: Specialization | ~3cm | 17cm |
| **Total** | **~14.5cm** | **17cm** |

---

## Figure Caption (Final Version)

**Fig. 1.** **Latency-tolerant cloud-edge collaborative VLA framework.** **(Top)** Deployment architecture. The cloud-side VLA backbone $f_\theta$ processes delayed observations $o_{t-k}$ to produce high-level planning features $h_{t-k}$. The edge-side VisionActionHead $g_\phi$ fuses these (potentially stale) features with real-time local vision features $z_t$ from the frozen encoder $v_\psi(o_t)$ to produce latency-robust actions. The cloud operates asynchronously; the edge never blocks. **(Middle)** Paired-frame dual-path training. Each training step samples a delayed frame $o_{t-d}$ and the current frame $o_t$ from the same episode window. Both are processed through the shared backbone, and both resulting planning features are fused with the shared real-time vision features $z_t$. The dual-path loss $\mathcal{L} = \mathcal{L}_{\text{fresh}} + \mathcal{L}_{\text{stale}}$ simultaneously trains the backbone to produce delay-invariant representations and the action head to compensate for stale features using vision. **(Bottom)** Emergent representational specialization. Standard VLA training encodes all task information—including timing-critical details—into the backbone, creating fragility under delay. Our training naturally separates time-invariant semantics (backbone) from time-critical visual-motor details (action head + vision encoder), yielding robustness without explicit constraints.

---

## Quick-Start: How to Draw This

1. **Start with Panel A** — draw the cloud box and edge box, add the network boundary
2. **Add Panel B below** — draw the window bar first, then the two parallel paths
3. **Add Panel C at the bottom** — simple two-column comparison
4. **Connect the panels** — use subtle visual cues (e.g., the VLA box in Panel A matches the VLA box style in Panel B)
5. **Add arrows and labels** — use the color palette above
6. **Add annotations last** — text callouts, gradient labels, badges
7. **Test in grayscale** — ensure all paths are distinguishable
