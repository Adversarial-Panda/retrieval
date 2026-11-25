# Training-free regional retrieval with DINOv2 and DINOv3

Experiment report, 2026-11-25. Every number comes from `data/<experiment>/summary.txt` and `results.json`, produced by
the standalone scripts `exp*.py` in this folder.

---

## 1. Summary

**Question.** Can a frozen self-supervised ViT (DINOv2, DINOv3) retrieve object instances better if each database image
also stores embeddings of a few object-like regions found by the model itself?

**Method in one paragraph.**

1. **Database side.** Group the patch tokens of a few chosen blocks with spherical k-means and with a modern Hopfield

   update whose temperature is calibrated per patch to a fixed perplexity. Every 8-connected piece of every group
    becomes a candidate box.
2. **Ranking and storage.** Rank the boxes by last-block CLS-attention mass, keep the top K after NMS, and store one

   CLS embedding per crop plus one for the whole image.
3. **Query side.** Embed the query as one view: the whole image, or its given box in the standard protocol.
4. **Scoring.** A database image scores the best cosine similarity over its stored views.

There is no training and no re-ranking.

**Main results** (Hard mAP; "full" = whole query image, "box" = query cropped by its given box, the standard revisitop
protocol):

| Backbone | Method | ROxford full | ROxford box | RParis full | RParis box |
|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | global CLS, 448 | 53.37 | 45.03 | 81.35 | 76.75 |
| DINOv2 ViT-B/14 | **+ regions, blocks [1, 10], K = 20** | **64.64** | **64.73** | **86.28** | **83.51** |
| DINOv3 ViT-B/16 | global CLS, 448 | 61.95 | 58.35 | 82.50 | 76.78 |
| DINOv3 ViT-B/16 | **+ regions, blocks [0, 9, 10], K = 25** | **73.91** | **74.78** | **86.16** | **84.57** |

**Key findings.**

1. **Large gains on ROxford.** Database regions add +11 to +20 Hard mAP over the global descriptor on ROxford, and

   +4 to +8 on RParis, for both backbones.
2. **Most of the gain is dense multi-view storage; localization adds the rest.** A regular R-MAC grid with the same

   number of regions already gets most of the way. Our proposals beat it by +0.7 to +5.1 Hard (DINOv2, significant
    in 3 of 4 settings), and beat randomly placed boxes of the same sizes everywhere.
3. **The design was chosen without tuning on the reported queries, as far as possible.**
   - Block sets and K were first screened on LVIS (object recall).
   - They were then selected on retrieval across datasets (choose on one, report on the other).
   - Two independently chosen DINOv2 designs give the same ladder (Section 4.4).
4. **The regime is consistent across backbones.** One early block plus one or two late blocks, K = 20–25, works best

   for both DINOv2 and DINOv3.
5. **Competitive with supervised methods, without training.** With DINOv3 and the standard box protocol, strictly

   held-out configurations reach **72.5 / 84.6** Hard (ROxford / RParis). Supervised RN101-SuperGlobal without
    re-ranking reports 72.1 / 83.5. The costs are about 26 stored vectors per image, and these two datasets may
    overlap with the DINO training data (Section 7).

---

## 2. Setup

| Item | Value |
|---|---|
| Backbones (frozen) | DINOv2 ViT-B/14 with 4 registers (`weights/dinov2_vitb14_reg4_pretrain.pth`); DINOv3 ViT-B/16 with 4 registers, RoPE (`weights/dinov3-vitb16-pretrain-lvd1689m`, Hugging Face format) |
| Retrieval data | ROxford5k (4,993 database images, 70 queries) and RParis6k (6,322 / 70), revisited ground truth |
| Validation data | LVIS v1 val, the 4,752 labeled images that are COCO val2017 images (50,672 objects, 1,203 categories) |
| Metric | revisited mAP, Medium (easy + hard positive) and Hard (hard positive, easy ignored); revisitop's `compute_ap` reimplemented and unit-tested |
| Statistics | paired bootstrap over the 70 queries (10,000 resamples): mean Hard-AP difference with 95% CI |
| Hardware | RTX 4070 Ti (GPU 1), fp16 autocast for forwards, fp32 for attention and grouping |
| Seeds | 0 everywhere (k-means initialization, random-position controls, bootstrap) |

**GPU pipeline** (shared by all scripts):

1. File bytes are read in DataLoader workers and decoded on the GPU with nvJPEG, once per image.
2. Resizing (bicubic, antialias), cropping and square padding run on the GPU.
3. Forwards are batched over images that share a resized shape.
4. k-means and Hopfield grouping are batched over images and blocks.
5. Connected pieces come from GPU label propagation, and NMS from `torchvision.ops.batched_nms`.

Proposals take about 22 ms per image (DINOv2, 2 blocks); embedding the crops is the dominant cost.

**Attention with RoPE (DINOv3).** The CLS-to-patch attention is recomputed by hand: the CLS query is not rotated, and
the patch keys are rotated with the model's own `apply_rotary_pos_emb`. Every DINOv3 script checks this against the
model before running. The CLS attention row matches to 2e-8, and the rebuilt last layer matches to 7e-4 (float32
rounding of the fused kernel).

---

## 3. Method details

**Database regions (final method):**

1. Resize to long side 448 (sides rounded to multiples of the patch size).
2. Take L2-normalized patch tokens after each chosen block.
3. Per block, group the patches with spherical k-means (k = 4, 8) and with a modern Hopfield update

   `ξ ← normalize(softmax(β_i ξ Xᵀ) X)` run for 10 steps from every patch. β_i is set per patch by binary search so
    that the softmax has perplexity 10 or 30 ("effective neighbours"). Patches whose fixed points have cosine > 0.9
    share a group.
4. Every 8-connected piece of every group becomes a box, grown by 10% per side and at least 64 px.
5. Keep boxes covering 5–90% of the image.
6. Score each box by the last-block CLS-attention mass inside its piece, apply NMS 0.7, and keep the top K.
7. Crop each box from the original image, pad it to a square with the ImageNet mean colour, resize to 224, and embed

   it as a normalized CLS.
8. Store the K crop vectors plus one whole-image view.

**Query:** one view (the whole image, or its given box), square-padded and embedded the same way.

**Score:** for each database image, the maximum cosine over its views.

**Why perplexity calibration.** With a fixed β, every layer jumps from "one group" to "hundreds of groups" at a
different β (preliminary study, Section 6). Calibrating β per patch to an effective number of neighbours makes the
grouping granularity mean the same in every block.

---

## 4. Experiments

### 4.1 exp1: global baseline (`exp1.py`, `exp1_dinov3.py`)

Every image is resized to long side 224 and 448 and embedded as one normalized CLS vector; ranking is by cosine.

| Backbone | Size | ROxford full M / H | ROxford box M / H | RParis full M / H | RParis box M / H |
|---|---|---|---|---|---|
| DINOv2 | 224 | 74.27 / 51.18 | 67.29 / 46.00 | 90.58 / 79.14 | 88.37 / 75.22 |
| DINOv2 | 448 | 75.44 / 53.37 | 67.68 / 45.03 | 91.79 / 81.35 | 89.31 / 76.75 |
| DINOv3 | 224 | 80.14 / 58.37 | 74.21 / 54.82 | 90.51 / 79.17 | 88.50 / 75.48 |
| DINOv3 | 448 | 81.47 / 61.94 | 76.25 / 58.35 | 92.05 / 82.50 | 89.25 / 76.78 |

**Findings:**

- The DINOv2 224 row reproduces the published DINOv2 ViT-B/14 numbers (72.9 / 49.5 / 90.3 / 78.6 without registers)
with *full* queries. A protocol study (Section 6) shows that the published numbers correspond to uncropped queries.
- 448 helps both backbones: about +2.2 Hard with DINOv2 and +1.3 to +3.6 with DINOv3. DINOv3 also improves box
queries at 448 (+3.5 [+1.6, +5.5]), where DINOv2 does not.
- DINOv3's global descriptor is far stronger on ROxford (+8.6 / +13.3 Hard at 448) and equal on RParis.

### 4.2 exp2: screening block sets × K on LVIS (`exp2.py`, `exp2_dinov3.py`)

**Purpose.** Choose which blocks the proposals come from, and how many regions K to keep, on validation data instead
of the retrieval test queries.

**Procedure:**

- Candidates from all 12 blocks are computed once per LVIS image, with exactly the retrieval pipeline above.
- Every block set of 1–3 blocks (298 sets) is scored at K ∈ {3, 5, 10, 15, 20, 25, 30}. The score is **recall@K**:
the share of labeled objects matched (box IoU ≥ 0.5) by one of the top K boxes.
- Recall always grows with K, so sets are compared only within the same K. The **5 best sets per K** become
candidates, 35 (set, K) combinations per backbone.
- A random-position control (the same boxes at random places) is reported for every set.

**Top 2 sets per K (recall@K, all objects; the random control is about half of each value):**

| K | DINOv2 | DINOv3 |
|---|---|---|
| 3 | [7] 4.9, [10] 4.9 | [11] 5.4, [10] 5.3 |
| 5 | [10] 6.4, [7] 6.3 | [10] 6.9, [11] 6.8 |
| 10 | [10] 8.8, [9] 8.7 | [10] 9.5, [9, 10] 9.2 |
| 15 | [0, 10] 10.3, [7, 10] 10.3 | [10] 10.9, [9, 10] 10.8 |
| 20 | [0, 10] 11.4, [7, 10] 11.4 | [8, 10] 11.8, [9, 10, 11] 11.8 |
| 25 | [0, 10] 12.3, [0, 7, 10] 12.3 | [0, 10] 12.5, [0, 9, 10] 12.4 |
| 30 | [0, 9, 10] 13.0, [0, 7, 10] 13.0 | [0, 9, 10] 13.0, [0, 8, 10] 13.0 |

**Findings:**

- **The number of useful blocks grows with K:** single late blocks win at K ≤ 5–10, block 10 plus a partner at
K = 15–20, and an early block (0) plus late blocks at K = 25–30.
- **A single block runs out of distinct regions.** For example, DINOv2 block 10 stays at 11.1 from K = 25 on, while
[0, 10] keeps rising to 12.9.
- **DINOv3's single blocks are stronger** (block 10: 10.9 vs 10.1 at K = 15), so its late blocks carry further
before early blocks help.
- **Within a K, the top sets are tied** (spread from #1 to #8: 0.06–0.59 points). LVIS fixes the *regime* but cannot
pick the winner, which is why exp3 decides on retrieval.

### 4.3 exp3: selecting the design on retrieval, across datasets (`exp3.py`, `exp3_dinov3.py`)

**Procedure:**

- The 35 combinations from exp2 run on ROxford and RParis in one GPU pass. Every unique box is embedded once, and
smaller K are prefixes of each set's top regions.
- **Rule, fixed in advance:** maximize Hard mAP with full queries on the *selection* dataset. Configurations within
0.5 points of the best count as tied; ties go to the smaller K, then to fewer blocks.
- The rule is applied in both directions: choose on RParis and report ROxford, and the reverse.

**Best per K (Hard, full / box):**

| K | DINOv2 ROxford | DINOv2 RParis | DINOv3 ROxford | DINOv3 RParis |
|---|---|---|---|---|
| 3 | 57.59 / 55.49 | 83.11 / 78.82 | 67.17 / 66.04 | 83.24 / 79.89 |
| 5 | 59.52 / 58.03 | 84.26 / 80.51 | 68.68 / 67.26 | 84.14 / 81.17 |
| 10 | 61.61 / 59.86 | 85.44 / 82.05 | 71.92 / 71.58 | 85.40 / 83.53 |
| 15 | 63.22 / 63.50 | 86.12 / 83.25 | 72.68 / 72.56 | 85.93 / 84.13 |
| 20 | 64.64 / 64.73 | 86.36 / 83.58 | 73.17 / 73.56 | 85.98 / 84.38 |
| 25 | 64.74 / 64.78 | 86.76 / 84.16 | 73.91 / 74.78 | 86.16 / 84.57 |
| 30 | 64.79 / 64.75 | 87.00 / 84.57 | 74.00 / 75.00 | 86.40 / 84.72 |

**Cross-dataset selection (strictly held-out results):**

| Backbone | Chosen on | Configuration | Held-out result (Hard, full / box) |
|---|---|---|---|
| DINOv2 | RParis | blocks [0, 7, 11], K = 25 | ROxford 64.37 / 65.29 |
| DINOv2 | ROxford | blocks [1, 10], K = 20 | RParis 86.28 / 83.51 |
| DINOv3 | RParis | blocks [7, 10], K = 15 | ROxford 72.00 / 72.53 |
| DINOv3 | ROxford | blocks [0, 9, 10], K = 25 | RParis 86.16 / 84.57 |

**Frozen designs.** The same rule applied to the average over both datasets gives:

- **DINOv2: blocks [1, 10], K = 20** (average 75.46; best 75.84);
- **DINOv3: blocks [0, 9, 10], K = 25** (average 80.04; best 80.09).

These freezes use both test sets, so they serve as the design for the next steps. Their unbiased numbers are the
held-out rows above.

**Findings:**

- Retrieval keeps improving up to K = 20–25 and then levels off.
- At large K, an early block plus late blocks beats pairs of late blocks, especially on ROxford. For example, DINOv2
at K = 20: [1, 10] 64.64 vs [7, 10] 62.47.
- The two directions pick different exact configurations but the same regime. Within a K, differences are about
1–2 points on ROxford, close to the implementation-level noise (about 1.5 on ROxford).

### 4.4 exp4: the ladder from baseline to final method (`exp4.py`, `exp4_dinov3.py`)

Each row changes one thing. Rows C–E store crops plus the whole view, use a one-view query and score by max-sim. C,
D and E each have a **random-position control**: the same boxes moved to uniformly random positions (D's control is
the first 3 of E's).

| Row | Change |
|---|---|
| A | global CLS, 224 |
| B | + resolution 448 |
| C | + database regions from the CLS-attention heatmap, K = 3 (smallest patch set holding 60% of the attention, 8-connected blobs ranked by mass, whole-image fallback) |
| D | heatmap blobs → patch-feature proposals from the frozen blocks, K = 3 |
| E | proposals K = 3 → frozen K (final) |

**DINOv2, blocks [1, 10], K = 20** (Hard; Medium in `data/exp4/summary.txt`):

| Row | ROxford full | ROxford box | RParis full | RParis box |
|---|---|---|---|---|
| A | 51.16 | 45.99 | 79.14 | 75.22 |
| B | 53.37 | 45.03 | 81.35 | 76.75 |
| C | 56.93 | 53.20 | 82.50 | 78.52 |
| D | 56.52 | 52.51 | 82.69 | 78.49 |
| **E** | **64.64** | **64.73** | **86.28** | **83.51** |
| E, random positions | 60.71 | 58.36 | 84.89 | 80.96 |

**DINOv3, blocks [0, 9, 10], K = 25:**

| Row | ROxford full | ROxford box | RParis full | RParis box |
|---|---|---|---|---|
| A | 58.37 | 54.82 | 79.17 | 75.48 |
| B | 61.95 | 58.35 | 82.50 | 76.78 |
| C | 66.31 | 64.35 | 83.38 | 79.23 |
| D | 65.28 | 63.14 | 82.76 | 79.00 |
| **E** | **73.91** | **74.78** | **86.16** | **84.57** |
| E, random positions | 70.84 | 71.51 | 85.22 | 83.21 |

**Step differences** (Hard, 95% CI, full query; box query in the summaries):

| Step | DINOv2 ROxford | DINOv2 RParis | DINOv3 ROxford | DINOv3 RParis |
|---|---|---|---|---|
| B vs A | +2.2 [+0.7, +3.6] | +2.2 [+1.1, +3.4] | +3.6 [+1.6, +5.7] | +3.3 [+2.3, +4.5] |
| C vs B | +3.6 [+1.3, +5.7] | +1.2 [+0.0, +2.3] | +4.4 [+1.9, +6.9] | +0.9 [−0.3, +2.0] |
| D vs C | −0.4 [−1.4, +0.5] | +0.2 [−0.4, +0.8] | −1.0 [−2.3, +0.4] | −0.6 [−1.4, +0.3] |
| **E vs D** | **+8.1 [+5.9, +10.3]** | **+3.6 [+1.9, +5.6]** | **+8.6 [+6.3, +11.1]** | **+3.4 [+2.3, +4.5]** |
| **E vs A** | **+13.5** | **+7.1** | **+15.6** | **+7.0** |

**E vs its random-position control** (Hard, full / box):

| | ROxford | RParis |
|---|---|---|
| DINOv2 | +3.9 [+2.4, +5.5] / +6.4 [+3.9, +9.3] | +1.4 [+0.8, +2.1] / +2.6 [+1.6, +3.6] |
| DINOv3 | +3.1 [+2.0, +4.2] / +3.3 [+1.9, +4.6] | +0.9 [+0.7, +1.2] / +1.4 [+1.0, +1.8] |

**Robustness: the other DINOv2 cross-dataset choice** (`data/exp4_b0-7-11_k25`, blocks [0, 7, 11], K = 25):

| | ROxford full | ROxford box | RParis full | RParis box |
|---|---|---|---|---|
| E, [0, 7, 11], K = 25 | 64.37 | 65.29 | 86.76 | 84.16 |
| Δ vs [1, 10], K = 20 | −0.3 [−1.0, +0.5] | +0.6 [−0.3, +1.6] | +0.5 [+0.3, +0.7] | +0.7 [+0.4, +1.0] |

Each design is better only on the dataset it was chosen on, and tied on the other. That is the expected signature of
selection noise, not a real difference. [1, 10], K = 20 is kept because it is cheaper (2 blocks instead of 3, 21
stored vectors per image instead of 26, 40% faster).

**Findings:**

- **Every step except D is significant on ROxford for both backbones.** D (heatmap → proposals at equal K = 3) is a
tie or slightly worse: proposals only pay off once K is large (E vs D is +8 Hard on ROxford).
- **At the final design, box placement matters everywhere:** E beats its random-position control in all 8 settings
(2 backbones × 2 datasets × 2 protocols).
- **DINOv3 raises everything on ROxford** (final 73.9 vs 64.6), and both backbones level off around 86 Hard
(full query) on RParis.

### 4.5 exp5: grid-region baseline (`exp5.py`, DINOv2)

**Purpose.** Is the gain from finding objects, or would any dense set of regular crops do as well?

**Procedure:** the pipeline of exp4 row E, with only the source of the boxes changed:

- **grid:** R-MAC regions (Tolias et al., in the layout of cirtorch's `rmac`): squares of side 2·min(W, H)/(l+1) at
scales l = 1, 2, …, about 40% overlap, coarse first. Each image gets as many grid regions as it has proposals
(2 + 6 + 12 = 20 for a 4:3 image).
- **grid@K:** the first K grid regions, K ∈ {5, 10, 20, 30}.
- **random:** our boxes at random positions.

| Database regions (+ whole view) | ROxford full | ROxford box | RParis full | RParis box |
|---|---|---|---|---|
| B: global 448 | 53.37 | 45.03 | 81.35 | 76.75 |
| random positions | 59.48 | 57.41 | 85.30 | 81.77 |
| grid@5 | 57.91 | 54.46 | 83.33 | 79.07 |
| grid@10 | 60.18 | 56.77 | 84.29 | 80.11 |
| grid@20 (matched count) | 62.48 | 59.66 | 85.60 | 81.89 |
| grid@30 | 63.31 | 61.78 | 86.43 | 83.24 |
| **ours, K = 20** | **64.64** | **64.73** | 86.28 | **83.51** |

| Hard difference [95% CI] | ROxford full | ROxford box | RParis full | RParis box |
|---|---|---|---|---|
| **ours vs grid, same count** | +2.2 [−0.1, +4.2] | **+5.1 [+3.4, +6.8]** | **+0.7 [+0.1, +1.4]** | **+1.6 [+0.6, +2.7]** |
| ours vs grid@30 | +1.3 [−0.9, +3.4] | **+2.9 [+1.6, +4.3]** | −0.2 [−0.6, +0.3] | +0.3 [−0.3, +0.9] |
| grid vs random positions | **+3.0 [+1.5, +4.8]** | **+2.3 [+0.5, +4.2]** | +0.3 [−0.2, +0.7] | +0.1 [−0.5, +0.8] |

**Findings:**

- **Ours beats the grid at equal storage** in all four settings: significantly in three, borderline in ROxford full.
The gap is largest on ROxford with box queries, where the query frames an object and our regions do too.
- **The grid needs 50% more regions to come close,** and still loses on ROxford box (+2.9). Ours gives the same or
better accuracy with a third less storage.
- **Most of the gain is dense multi-view storage.** Of +11.3 over B on ROxford full, a grid already gets +9.1. The
claim must be stated as "localization adds +0.7 to +5.1 over the strongest regular-crop baseline", not as the whole
gain.
- **The grid is a stronger control than random boxes** on ROxford.

---

## 5. Comparison with published methods

Standard box-query protocol, Hard mAP, without 1M distractors. Published rows are from the SuperGlobal paper (ICCV
2023, Table 1).

| Method | Training | Vectors per image | ROxford H | RParis H |
|---|---|---|---|---|
| RN101-DELG, global | supervised (landmarks) | 1 | 55.6 | 72.4 |
| RN101-DOLG | supervised | 1 | 61.1 | 80.3 |
| RN101-CVNet, global | supervised | 1 | 63.1 | 79.1 |
| RN101-SuperGlobal, no re-ranking | supervised | 1 | 72.1 | 83.5 |
| RN101-SuperGlobal, re-ranking top 400 | supervised | 1 (+ re-ranking) | 80.2 | 86.7 |
| DINOv2 + regions (frozen design) | none | 21 | 64.7 | 83.5 |
| DINOv3 + regions, held-out configuration | none | 16–26 | 72.5 | 84.6 |
| DINOv3 + regions (frozen design) | none | 26 | 74.8 | 84.6 |

Without any training, DINOv3 + regions matches supervised SuperGlobal without re-ranking on ROxford and exceeds it on
RParis. Two costs come with that: more stored vectors per image, and no evaluation yet with 1M distractors.

---

## 6. Earlier findings that shaped the method

These come from the preliminary series (scripts in `tools/`, report in `tools/REPORT.md`; their result folders were
removed when the series was restarted).

- **Protocol.** The published DINOv2 numbers are reproduced only with *uncropped* queries; box-cropped queries are
6–8 Hard lower on ROxford for the global descriptor.
- **Query side.** Localizing the query does not help. Even an oracle query crop (best match to the given box) is
about 5 Hard below the whole query on ROxford. Keep the query whole and localize the database.
- **One region per image is mostly a multi-view effect.** K = 1 gains came almost entirely from adding a second view
of the image; localization effects appear only with several regions.
- **CLS attention covers one salient region.** Patch features separate objects (an arm and hand, a sleeve, donuts)
that CLS attention ignores; this motivated patch-feature proposals.
- **Fixed-β Hopfield grouping fails per layer.** A layer goes from 1 group to hundreds within a factor of 2 in β, at
a different β per layer. Perplexity calibration fixes it.
- **Patches per object, not the layer, limit small objects.** At 448 px a patch-aligned box can match only 22% of
COCO small objects; at 2× upsampling, 80%.
- **Implementation sensitivity.** Switching from PIL to GPU decoding and resizing moved ROxford Hard by about 1.5
points with no change in the method, so all compared rows in this report come from one implementation.

---

## 7. Caveats

1. **Possible training contamination.** DINOv2's training set was built partly by retrieving images similar to the

   ROxford and RParis databases (DINOv2 paper, Table 15). DINOv3 uses a different, larger pipeline that was not
    checked here. Results on these two datasets may be optimistic for both backbones; **ILIAS is the required
    independent test.**
2. **Design selection.** Screening (LVIS) and selection (cross-dataset) limit tuning on the reported queries, but

   the frozen designs use both test sets. The strictly held-out numbers are in Section 4.3.
3. **70 queries per dataset.** Differences below about 1–2 Hard points are within noise; the bootstrap CIs are the

   reference.
4. **Storage.** 21–26 vectors per database image instead of 1. With 1M distractors that is about 21–26M vectors, so

   compression (PQ) and the distractor effect are untested.
5. **Grid baseline only for DINOv2.** The DINOv3 grid comparison (exp5_dinov3) has not been run yet.
6. **ViT-B only.** Larger backbones (ViT-L) are untested.

---

## 8. Next steps

1. **exp5_dinov3:** grid baseline for DINOv3, to see whether localization still beats a regular grid with the

   stronger backbone.
2. **ILIAS** with both frozen designs (DINOv2 [1, 10], K = 20; DINOv3 [0, 9, 10], K = 25): the uncontaminated,

   held-out test.
3. **+R1M distractors** with PQ-compressed crop vectors: mAP vs bytes per image.
4. **ViT-L** for both backbones, for the final numbers.

---

## 9. Reproducing

All scripts are standalone (they share no code), run on GPU 1 by default, and write to `data/<name>/`
(`summary.txt`, `results.json`, `run.log`).

```bash
python exp1.py                  # DINOv2 global baseline              -> data/exp1
python exp2.py                  # DINOv2 LVIS screening, 35 configs   -> data/exp2 (top_configs.json)
python exp3.py                  # DINOv2 cross-dataset selection      -> data/exp3 (selection.json)
python exp4.py                  # DINOv2 ladder, blocks 1 10, K 20    -> data/exp4
python exp4.py --blocks 0 7 11 --k 25 --compare data/exp4/results.json --out data/exp4_b0-7-11_k25
python exp5.py                  # DINOv2 grid baseline                -> data/exp5

python exp1_dinov3.py           # DINOv3 global baseline              -> data/exp1_dinov3
python exp2_dinov3.py           # DINOv3 LVIS screening               -> data/exp2_dinov3
python exp3_dinov3.py           # DINOv3 cross-dataset selection      -> data/exp3_dinov3
python exp4_dinov3.py           # DINOv3 ladder, blocks 0 9 10, K 25  -> data/exp4_dinov3
```

| Script | Runtime (GPU 1) |
|---|---|
| exp1 / exp1_dinov3 | about 1.5 min |
| exp2 / exp2_dinov3 | about 8 / 5.5 min |
| exp3 / exp3_dinov3 | about 36 / 25 min |
| exp4 / exp4_dinov3 | about 15.5 / 18 min |
| exp5 | about 21 min |

Data paths are set by `--root` (revisitop) and `--lvis` (LVIS); weights by `--weights`.
