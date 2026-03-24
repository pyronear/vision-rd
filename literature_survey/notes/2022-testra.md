# Real-time Online Video Detection with Temporal Smoothing Transformers

**Authors:** Zhao & Krähenbühl
**Year/Venue:** 2022, ECCV
**PDF:** `pdfs/2022-TeSTra-Temporal-Smoothing-Transformer-Online-Detection-Zhao-Krahenbuhl.pdf`

## Key Idea

Reformulates cross-attention in video transformers using temporal smoothing kernels, enabling O(1) per-frame updates for streaming video. Two kernels: Box (FIFO, O(N) memory) and Laplace (exponential smoothing, O(1) memory). This makes long-term temporal context free at inference time -- runtime is constant regardless of how much history is considered.

## Architecture

```
 Live streaming video
 ───────────────────────────────────────────▶ time
 │         Long-term memory          │ Short │
 │ (all past frames, compressed)     │ term  │
 │                                   │(L=32) │
 └───────────────┬───────────────────┴───┬───┘
                 │                       │
                 ▼                       │
          ┌─────────────┐               │
          │   Encoder   │               │
          │             │               │
          │ M=16 learned│               │
          │ queries     │               │
          │      │      │               │
          │      ▼      │               │
          │ ES-Attention│  O(1)/frame   │
          │ (Laplace    │  exponential  │
          │  kernel)    │  smoothing    │
          │      │      │               │
          │      ▼      │               │
          │ l_enc cross │               │
          │ attn layers │               │
          └──────┬──────┘               │
                 │ M latent vectors Z   │
                 │ (compressed history) │
                 │                      │
                 └──────────┬───────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │   Decoder   │
                     │             │
                     │ Self-attn   │  on short-term
                     │      │      │  memory (L frames)
                     │      ▼      │
                     │ Cross-attn  │  attend to
                     │ [Z ∥ short] │  [compressed long
                     │             │   + short-term]
                     │      │      │
                     │ l_dec layers│
                     └──────┬──────┘
                            │
                            ▼
                     ┌─────────────┐
                     │ Linear +    │
                     │ Classifier  │  per-frame scores
                     └─────────────┘
```

## Two Temporal Kernels

```
Box Kernel (FIFO):  K_B(ω_t, ω_n) = 1[t-n < N]
════════════════
 ψ(t) = ψ(t-1) + κ(q,k_t) - κ(q,k_{t-N})
 φ(t) = φ(t-1) + κ(q,k_t)·v_t - κ(q,k_{t-N})·v_{t-N}

 Runtime: O(MC) per frame    ← constant
 Memory:  O(N)               ← must store window

 ┌─────────────────────────┐
 │ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■    │  fixed window of N
 │ ⊖                   ⊕   │  subtract oldest,
 │ (dequeue)      (enqueue)│  add newest
 └─────────────────────────┘


Laplace Kernel (Exponential Smoothing):  K_L(ω_t, ω_n) = e^{-λ(t-n)}
═══════════════════════════════════════
 ψ̃(t) = e^{-λ} · ψ̃(t-1) + κ(q,k_t)
 φ̃(t) = e^{-λ} · φ̃(t-1) + κ(q,k_t)·v_t

 Runtime: O(MC) per frame    ← constant
 Memory:  O(1)               ← only φ̃, ψ̃ scalars!

 λ controls decay speed:
   small λ → slow decay → long memory
   large λ → fast decay → short memory

 ●━━━━━━━━━━━━━━━━━━━━━━● new frame
 ▕▏            ▕▏       ▕▏
 old frames    recent   current
 (faded)       (strong) (full weight)
```

## Training vs Inference

- **Training:** windowed attention in matrix form (Eq. 9) with Vandermonde matrix encoding exponential decay. Standard GPU parallelism.
- **Inference:** recursive formulation (Eq. 6) for O(1) streaming.
- Choose λ and N such that `e^{-λ(N-1)}` is very small → windowed training ≈ infinite streaming. Verified: <0.05% difference (Table 4).

## Results

**THUMOS'14 online action detection:**

| Method | mAP   |
|--------|-------|
| OadTR  | 65.2  |
| LSTR   | 69.5  |
| **TeSTra** | **71.2** |

**Runtime (Fig 6, Table 7):**

| Component       | FPS    |
|-----------------|--------|
| TeSTra alone    | 142.8  |
| + NVOFA opt flow| 41.1   |
| + TV-L1 opt flow| 12.6   |

LSTR runtime grows linearly with memory length. TeSTra stays flat at 142.8 FPS regardless of N.

**Ablations:**
- Laplace > Box kernel (16.95 vs 16.14 action recall)
- Positional embedding NOT needed for long-term memory
- PE needed for short-term memory (-1.3% without)
- MixClip augmentation: +1.5%
- Best fusion: concatenate compressed long-term + short-term (17.0%)

## Applicability to Pyronear

**High relevance for the streaming/online detection aspect.**

**Why TeSTra fits Pyronear:**
1. **O(1) cost per frame** -- process every 30s frame indefinitely without growing compute or memory.
2. **λ parameter directly controls memory timescale.** For smoke detection, set λ small (slow decay) so frames from 5-30 minutes ago still contribute. Smoke evolves over minutes; λ tuning is the key.
3. **No positional embedding for long-term memory** -- the model doesn't care about exact frame timing, just content and recency via exponential decay. This is perfect for 30s intervals where absolute timestamps don't matter.
4. **Short-term + long-term memory** maps to: "is something new appearing right now?" (short-term, last 2-3 frames) + "has this area been clear for the past hour?" (long-term, exponentially compressed).

**Adaptations needed:**
- Replace optical flow input with CNN features or background subtraction maps from fixed camera.
- Short-term memory L=32 frames → L=4-8 frames at 30s intervals (2-4 min window).
- Retune λ for 30s frame rate (original was tuned for 24fps video).
- The feature extractor (ResNet-50 two-stream) should be replaced with a single-stream CNN on the YOLO crop + change map.

**Server-side GPU deployment:** TeSTra at 142.8 FPS means the temporal model is never the bottleneck. The CNN feature extractor is the bottleneck (150 FPS for RGB). With 30s intervals, even a slow feature extractor has 30 seconds to process each frame -- no latency concern at all.

## Takeaways for Implementation

1. **Exponential smoothing attention is the key idea.** Even without the full TeSTra architecture, adding exponential smoothing to a simple temporal attention model would capture the core benefit.
2. **λ is the only critical hyperparameter** for temporal modeling. Start with λ that gives a half-life of ~5-10 minutes (at 30s intervals, this means λ ≈ 0.03-0.07 per frame).
3. **MixClip augmentation** is useful: compose long-term history from different cameras/sequences during training to prevent overfitting to scene-specific patterns.
4. **Two-memory design** (compressed long-term + recent short-term) is more principled than a flat sliding window.
