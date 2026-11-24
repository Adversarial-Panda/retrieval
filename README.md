# Crop-and-Embed Retrieval with DINOv2: Experiment Report

Experiments run 2026-11-24, following the roadmap in `docs/Crop-and-Embed Retrieval with DINOv2.md`.
Model: DINOv2 ViT-B/14 with 4 registers (`dinov2_vitb14_reg4_pretrain.pth`), frozen. All numbers come from the run folders under `runs/`.

---

## 1. Summary

**Question.** Does cropping the objects that DINOv2 finds, and embedding the crops, beat the plain DINOv2 global descriptor for instance retrieval?

**Answer.** Yes, if the crops are taken on the **database side only** and come from **patch-feature proposals** (k-means and Hopfield grouping), not from CLS attention. The query stays whole.

| Method (full-query protocol) | ROxford M | ROxford H | RParis M | RParis H |
|---|---|---|---|---|
| E1: global CLS at 448 (baseline) | 75.45 | 53.37 | 91.80 | 81.38 |
| **Database: top-15 proposals + whole view; query: whole image** | **79.38** | **61.93** | **93.97** | **86.27** |
| Gain | +3.9 | **+8.6** | +2.2 | **+4.9** |

On the same comparison, Hard mAP gains over E1 have 95% bootstrap CIs of ROxford [+5.0, +12.0] and RParis [+2.6, +7.5]. Against the same number of crops placed at random positions (with the same box sizes), the gain is +3.4 [+2.0, +4.9] on ROxford and +1.9 [+1.2, +2.8] on RParis. So localization explains part of the gain and multi-view coverage explains the rest.

**Main findings, in the order they were found.**

1. **The pipeline is correct.** The hand-recomputed attention matches the fused block (max abs diff 1.6e-4), and boxes line up with the images.
2. **CLS attention finds one salient region, not objects.** Its boxes are 1.5–2× too large, and it misses small objects and secondary objects. Intermediate-layer CLS attention does not fix this.
3. **Patch features separate objects; CLS attention does not.** k-means on patch features from blocks 7–11 separates the arm, the sleeve and the donuts in an image where CLS attention sees only donuts.
4. **For small objects, the patch grid is the limit, not the layer.** At 448 px, a patch-aligned box can match only 22% of COCO small objects at IoU ≥ 0.5. At 2× upsampling, 80% become reachable.
5. **A fixed β is the wrong control for a Hopfield update; calibrate per patch to a perplexity.** With a fixed β, every layer shows a phase transition at a different β. A per-patch β set to perplexity k (as in t-SNE) makes "k neighbours" mean the same thing in every layer.
6. **Ranking is the bottleneck on COCO.** With a perfect ranker, the top 5 of the same proposals would contain 3.3× more objects than with the CLS-attention ranker.
7. **The published DINOv2 numbers match full, uncropped queries.** Cropping queries by their box, which is the standard protocol, lowers ROxford mAP by 6–8 points.
8. **K = 1 crop gains are mostly a multi-view ensemble.** A whole-image second view gives most of the gain. K = 3 attention crops give a localization effect that passes controls (+4.8 ROxford H against the strongest baseline without localization).
9. **Keep the query whole and localize the database.** Even an oracle query crop does not beat using the whole query.
10. **k-means/Hopfield proposals beat attention crops on the database side.** The gain grows up to K = 15–20. The share due to localization peaks at K = 10–15 and fades by K = 30, where crops cover the image densely.

**Roadmap gates.**
- **Gate 1 passes**, but only in the full-query setting. In the standard cropped-query protocol, E0 is far below the published numbers (Section 4.2).
- **Gate 2 passes** in both protocols: at least +1 mAP on ROxford Hard over E1, including against controls without localization.

---

## 2. Setup

| Item | Value |
|---|---|
| Backbone | DINOv2 ViT-B/14 + 4 registers, built from the repo's `vit_base` with torch.hub settings (`interpolate_antialias=True`, `interpolate_offset=0`). Non-register model used only to check published numbers. |
| Weights | `weights/dinov2_vitb14_reg4_pretrain.pth` (official, dl.fbaipublicfiles.com); `weights/dinov2_vitb14_pretrain.pth` |
| Hardware | 2× RTX 4070 Ti (12 GB), fp16 autocast, attention maps in fp32 |
| Localization data | COCO val2017 (5,000 images, 36,335 non-crowd boxes from YOLO labels; 48 images have no boxes) |
| Retrieval data | ROxford5k (4,993 database / 70 queries), RParis6k (6,322 / 70), revisited ground truth (`gnd_*.pkl`) |
| Metric | mAP with revisitop's `compute_ap`/`compute_map`, reimplemented line for line and unit-tested. Medium: easy + hard are positives. Hard: hard are positives, easy are junk. |
| Statistics | Paired bootstrap over the 70 queries (5,000–10,000 resamples), reporting mean Hard-AP difference and 95% CI |
| Seeds | 0 everywhere (sampling, k-means initialization, random-box controls) |

**Data checks.**
- No COCO val2017, ROxford or RParis image carries an EXIF rotation. COCO labels and revisitop query boxes are therefore in stored-pixel coordinates.
- The retrieval loader deliberately does **not** apply EXIF transposition.
- The YOLO label count matches the non-crowd JSON annotations exactly (36,335).

**Code** (all in the project root):

| Script | Purpose |
|---|---|
| `localize.py` | Roadmap steps 1–4: loading, aspect-preserving resize to multiples of 14, padded native input, hand-recomputed CLS attention, mass and PCA masks, connected-component boxes with fallback, boxes back to pixels |
| `coco_loc.py` | COCO localization: one pass per input size caches per-layer/per-head CLS attention; then scores mask rules, K and margins |
| `probe_layers.py` | Patch-feature k-means proposals per layer and input size: upper-bound recall and the patch-grid limit |
| `hopfield_regions.py` | Hopfield-attractor grouping, SoftKNN self-retrieval contrast, ranking comparison, greedy oracle ranking |
| `retrieval.py` | Global descriptors (CLS/GeM), revisitop mAP, query/resize protocol options (E0/E1) |
| `crop_retrieval.py` | Crop-and-embed (E2/E3/E2b/E6) with box-source controls (`--boxes attention / center-matched / random-matched / center60 / full`) |
| `bootstrap_e6.py` | Paired bootstrap for E6/E7 against baselines without localization |
| `asymmetric.py` | Whole query against database crops |
| `query_select.py` | Query-side candidate crops and selection rules, including the oracle |
| `db_proposals.py` | Database-side k-means/Hopfield proposals, their embeddings and random-position controls |
| `db_eval.py`, `db_eval_k.py` | Database crop sets under one scoring rule, K sweep |

---

## 3. Part A: Object localization on COCO val2017

COCO was used first, as a fast and fully labeled check of roadmap steps 1–4 before any retrieval.

### 3.1 Pipeline (roadmap steps 1–4)

1. **Load:** RGB, EXIF orientation applied, original resolution kept.
2. **Resize:** long side to S, each side rounded to the nearest multiple of 14, bicubic with antialiasing. Scale factors `sx = W/w'`, `sy = H/h'` are recorded. Alternatively, **native + padding**: no resize; the right and bottom are padded to a multiple of 14 with the ImageNet mean, and boxes map back at scale 1.
3. **CLS attention:** run blocks 0..10 normally, then recompute the CLS query row of the chosen block's attention by hand (`softmax(q_cls·kᵀ·scale)`). Drop CLS and the 4 registers (`1 + R` tokens) and reshape to the `gh × gw` patch grid.
4. **Mask:**
   - **Method A:** average the heads, normalize, keep the smallest set of patches holding τ of the mass (τ ∈ {0.5, 0.6, 0.7}).
   - **Method B:** first PCA component of the last-layer patch features; the sign is chosen so the positive side holds more CLS attention.
5. **Boxes:** 8-connected components, ranked by attention mass. Components smaller than 1% of the grid are dropped, and the top K are kept. **Fallback** to the whole image when no component remains or the top box covers more than 90% of the grid.
6. **Back to pixels:** cell `(r, c)` covers `[c·14, (c+1)·14) × [r·14, (r+1)·14)`, scaled by `(sx, sy)`. Optional margin (0/10/20%) and minimum side; minimum side 0 for IoU scoring.

**Check of step 3.** The last block rebuilt from the hand-computed full attention matches the fused block: max abs diff 1.6e-4. The CLS output matches `forward_features`: 6e-6. The CLS row matches the full matrix: 1.4e-7.

**Metrics.**
- **CorLoc:** share of images whose top box hits some ground-truth object at IoU ≥ 0.5.
- **R@K:** per-object recall at IoU ≥ 0.5 with K candidates, per COCO size bin. Bins use box area: small < 32², medium < 96², large otherwise.
- **Largest-object hit:** top box hits the largest object.
- **Fallback rate.**

### 3.2 Mask rule and input size (`runs/coco_loc`, `runs/coco_loc_native`; 4,952 images with boxes; margin 0)

| Input | Rule | CorLoc | Mean IoU (top box) | Largest object | Fallback % | R@1 | R@5 | R@5 small / medium / large |
|---|---|---|---|---|---|---|---|---|
| any | whole-image prior | 23.3 | 0.324 | 23.3 | 0 | 3.5 | 3.5 | 0 / 0 / 10.5 |
| 224 | attention τ = 0.6 (roadmap default) | 35.8 | 0.403 | 27.0 | 0.2 | 5.2 | 6.5 | 0 / 1.2 / 18.2 |
| 224 | attention τ = 0.7 | 36.7 | 0.405 | 31.2 | 2.7 | 5.4 | 6.5 | 0 / 0.9 / 18.6 |
| 224 | PCA | 31.8 | 0.373 | 30.4 | 23.1 | 4.9 | 5.2 | 0 / 0.2 / 15.4 |
| 448 | attention τ = 0.6 | 35.1 | 0.409 | 23.8 | 0.2 | 5.0 | 6.5 | 0 / 2.0 / 17.3 |
| 448 | attention τ = 0.7 | 40.1 | 0.432 | 31.5 | 0.3 | 5.9 | 7.5 | 0 / 1.7 / 20.6 |
| 448 | PCA | 32.8 | 0.379 | 31.3 | 23.3 | 5.0 | 5.6 | 0 / 0.5 / 16.4 |
| 640 | attention τ = 0.7 | 41.3 | 0.438 | 31.4 | 0.2 | 6.0 | 7.7 | 0 / 1.7 / 21.4 |
| native + pad | attention τ = 0.7 | **41.5** | **0.440** | **32.1** | 0.2 | **6.1** | **7.7** | 0 / 1.5 / 21.5 |

Margins of 10% and 20% always lower IoU, because the boxes are already too large. Margin should be chosen by retrieval results, not by IoU.

**Diagnosis** (visual check plus measurements):
- **Boxes are too large.** The median top box is 1.97× (at 224) and 1.52× (at 448) the area of its best-matching object. Attention spills onto neighbouring objects and background, and multi-object scenes merge into one box.
- **The mask is not very diffuse.** It needs 13–19% of the grid at 224 and 9–13% at 448 to hold 50–70% of the mass. Single heads are sharper: the median head needs 7–12% for 60%.
- **Small objects are structurally impossible.** At 448, a 640 px image has a 32×24 grid, so the 1%-of-grid filter removes any component under about 8 patches (≈ 3,200 px²). A COCO small object is under 1,024 px².
- **PCA (Method B)** falls back to the whole image in 23% of images.

### 3.3 CLS attention from intermediate layers (`runs/coco_loc_layers`; 448, τ = 0.7)

| Block | 2 | 4 | 6 | 8 | 9 | 10 | 11 (last) |
|---|---|---|---|---|---|---|---|
| CorLoc | 27.6 | 28.3 | 28.7 | 31.7 | 35.4 | **42.1** | 40.1 |
| R@5 medium | 0.4 | 0.4 | 0.4 | 0.6 | 0.6 | 0.9 | **1.7** |
| Fallback % | 47.7 | 43.5 | 36.2 | 18.8 | 11.8 | 2.4 | 0.3 |

Earlier CLS attention is spread too thinly and triggers fallback often. No layer finds more small or medium objects. The signal is the problem, not the layer.

### 3.4 Patch features vs CLS attention (one image: `runs/coco_loc_layers/383337_layers.png`)

On 000000383337.jpg (donuts, an arm and hand, people), CLS attention at blocks 4, 8 and 11 never covers the arm. k-means (k = 8) on patch features from blocks 7, 9 and 11 gives clean, separate groups for the arm and hand, the sleeve, the legs, the plastic box and the donuts.

### 3.5 Which layer, which resolution? Upper-bound probe (`runs/probe_layers`, `runs/probe_native`, `runs/probe_native_vs_resize`)

Setup: 500 random COCO val images. k-means at k ∈ {4, 8, 16} on L2-normalized patch features. Every 8-connected piece of every cluster becomes a box, unranked, so this is an upper bound. "Grid ceiling" is the share of objects whose patch-snapped box reaches IoU ≥ 0.5, the best any patch-level method can do at that input size.

Recall % at IoU ≥ 0.5 (small / medium / large):

| Input | Grid ceiling | Block 2 | Block 5 | Block 8 | Block 11 | Stack 5+8+11 | Boxes/image (block 11) |
|---|---|---|---|---|---|---|---|
| 224 | 2.0 / 56.0 / 99.6 | 0.6 / 13.9 / 56.6 | 0.4 / 9.6 / 64.1 | 0.4 / 9.4 / 65.7 | 0.3 / 10.4 / 65.0 | 0.4 / 10.0 / 66.6 | 71 |
| 448 | 22.4 / 93.4 / 100 | 3.7 / 22.4 / 63.9 | 2.4 / 18.0 / 71.8 | 2.3 / 18.7 / 73.4 | 2.7 / 19.0 / 74.0 | 2.5 / 19.4 / 72.9 | 178 |
| 640 | 42.7 / 99.3 / 100 | 5.2 / 27.4 / 68.1 | 3.3 / 22.4 / 75.7 | 3.7 / 26.0 / 76.6 | 3.7 / 24.5 / 77.5 | 3.7 / 23.8 / 77.5 | 302 |
| native + pad | 39.6 / 99.1 / 100 | 5.8 / 26.2 / 67.0 | 4.4 / 20.6 / 73.6 | 4.1 / 22.7 / 75.7 | 3.3 / 22.4 / 76.1 | 3.2 / 22.5 / 77.0 | 296 |
| 896 (1.4× up) | 63.9 / 99.8 / 100 | 10.0 / 30.4 / 69.3 | 6.5 / 24.8 / 76.3 | 5.9 / 28.2 / 79.2 | 6.3 / 25.6 / 79.5 | 6.6 / 27.5 / 80.3 | 481 |
| 1280 (2× up) | 79.9 / 100 / 100 | 14.7 / 36.2 / 69.2 | 9.1 / 25.8 / 77.7 | 9.3 / 28.0 / 81.5 | 7.9 / 29.0 / 80.2 | 8.0 / 27.5 / 80.8 | 795 |

Findings:
- **ViTs keep the same 14 px grid in every layer.** Depth changes what the features mean, not their spatial resolution. Early blocks (2) separate small and medium objects slightly better by colour and texture, but break up large objects. Late blocks are best for large objects. The layers complement each other.
- **Concatenating layers averages the gains away.** Using several layers needs a union of per-layer proposals, not a stacked feature.
- **Patches per object matter, not pixel fidelity.** Upsampling COCO (at most 640 px) to 896 or 1280 keeps raising small-object recall, even though it adds no new detail.
- **Native + padding vs resize at the same scale is a tie.** Padding beats 224 and 448. Against 640 it is within about 1–2 points: better for small objects in blocks 2 and 5, worse in block 11 and for medium objects. Rounding to multiples of 14 changes the aspect ratio by under 1%, so resizing distorts almost nothing.

### 3.6 Combining layers with KNN / SoftKNN / Hopfield (`runs/hopfield_regions`)

The KNN, SoftKNN and modern Hopfield (MHN) classifiers from `main.py` were adapted to region finding. The memory is the image's own patch tokens at layer l.

**Grouping by Hopfield attractors.** Every patch runs the update `ξ ← normalize(softmax(β·ξ·Xᵀ)·X)` for 10 steps. Patches whose fixed points have cosine > 0.9 form one group, and each group is split into 8-connected pieces that become boxes.

*Problem found: a fixed β gives a phase transition at a different β in each layer.* Median number of groups on 10 images:

| Layer | β = 3 | β = 5 | β = 10 | β = 20 |
|---|---|---|---|---|
| 2 | 1 | 1 | 1 | 707 |
| 8 | 1 | 1 | 15.5 | 666 |
| 11 | 1 | 1 | 1 | 10.5 |

*Fix: calibrate β per patch* by binary search, so that `softmax(β_i · s_i)` has perplexity k, i.e. an effective k neighbours, as in t-SNE. KNN's k then becomes a granularity knob that means the same thing in every layer. Median number of groups:

| Layer | k = 10 | k = 30 | k = 100 |
|---|---|---|---|
| 2 | 5 | 2.5 | 1 |
| 8 | 30 | 6 | 2.5 |
| 11 | 10.5 | 4.5 | 2 |

**Ranking by self-retrieval.**
- *First version, purity:* the mass that a region's patches retrieve from the region, minus chance. It favoured tiny fragments, because neighbouring patches are near-duplicates.
- *Second version, contrast:* mass retrieved from inside the region minus mass retrieved from inside by the region's 1-patch ring. This is the roadmap's "local contrast" written in SoftKNN form. Layer combinations tried: own layer, mean, max, and vote over blocks 2, 5, 8 and 11.

Proposals on 500 images, native + padding, blocks 2/5/8/11. k-means uses k ∈ {4, 8, 16}; Hopfield uses perplexity ∈ {3, 10, 30}. Duplicates are removed with NMS at 0.7. Recall % at IoU ≥ 0.5:

| Proposal source | Boxes/image | Unranked recall (small / medium / large / all) |
|---|---|---|
| k-means | 1,308 | 10.9 / 47.2 / 91.7 / 50.1 |
| Hopfield | 1,995 | 10.7 / 30.6 / 59.5 / 33.6 |
| **both** | 3,303 | **16.5 / 55.1 / 92.4 / 55.0** |

| Ranking rule (k-means proposals), recall "all" | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| random | 0.2 | 0.5 | 1.0 | 2.1 |
| area (largest first) | 4.1 | 6.7 | 8.7 | 12.6 |
| **CLS attention mass** | **6.1** | **11.2** | **14.0** | 18.7 |
| contrast, own layer, k = 30 | 2.2 | 8.4 | 13.3 | **18.9** |
| contrast, max over layers, k = 30 | 2.0 | 8.6 | 13.4 | 18.9 |
| contrast, mean over layers, k = 30 | 3.2 | 7.8 | 11.7 | 16.7 |

Conclusions:
- Hopfield grouping is useful as a **second proposal source**. Together with k-means it adds +5.6 small and +7.9 medium recall.
- The hand-designed contrast score **does not beat CLS attention** as a ranker. Combining layers inside the score (max, mean, vote) was no better than using the region's own layer.
- No ranking rule places small objects in the top 10.

### 3.7 Oracle ranking upper bound (`runs/hopfield_oracle`)

The oracle picks the K proposals that cover the most objects, by greedy max coverage, which is guaranteed to reach at least 63% of the optimum. Both proposal sources (3,303 boxes per image). Recall % at IoU ≥ 0.5:

| K | Ranking rule | Small | Medium | Large | All |
|---|---|---|---|---|---|
| 1 | CLS attention | 0.0 | 0.0 | 12.1 | 3.8 |
| 1 | **oracle** | 1.7 | 9.5 | 38.0 | **16.1** |
| 3 | CLS attention | 0.0 | 0.0 | 31.1 | 9.8 |
| 3 | **oracle** | 7.2 | 26.8 | 70.7 | **34.6** |
| 5 | CLS attention | 0.0 | 0.5 | 41.7 | 13.3 |
| 5 | **oracle** | 11.4 | 39.8 | 82.3 | **44.5** |
| 10 | CLS attention | 0.1 | 3.9 | 52.4 | 18.0 |
| 10 | **oracle** | 15.7 | 53.2 | 91.2 | **53.7** |
| all | unranked | 16.5 | 55.1 | 92.4 | 55.0 |

At K = 3–5, ranking costs about 3.3× recall, mostly on medium objects (39.8 possible vs 0.5 achieved at K = 5). For small objects, the proposals themselves are the limit (16.5% even unranked). This motivates the roadmap's Experiment 3 verifier, preferably learned (group E) rather than hand-designed.

---

## 4. Part B: Retrieval on ROxford5k and RParis6k

### 4.1 Data and evaluation
- **Images:** downloaded directly from the official sources (Oxford VGG for images, CTU revisitop for ground truth). These are the same URLs the Hugging Face `galilai-group/revisitop` loader uses. Archives were resumable, checked against the server's file size, and extracted to `Dataset/revisitop/{roxford5k,rparis6k}/jpg/`.
- **Ground truth:** the `.pkl` files are loaded with a restricted unpickler that allows only numpy arrays and plain containers.
- **mAP:** revisitop's AP formula. The unit tests passed: a perfect ranking scores 1.0; positives at ranks 1 and 3 score 0.333; junk images are skipped correctly; queries with no positives are excluded; the Medium/Hard split is correct.

### 4.2 E0/E1: global descriptor and the Gate 1 protocol (`runs/E0`, `runs/E0_protocol`)

The DINOv2 paper (Table 9) reports ViT-B/14 at **72.9 / 49.5 / 90.3 / 78.6** (ROxford M/H, RParis M/H), but does not state its retrieval protocol. Resolution, query cropping and resize rule were swept on the model without registers (CLS descriptor):

| Protocol (no registers) | Ox M | Ox H | Par M | Par H |
|---|---|---|---|---|
| **Published** | 72.9 | 49.5 | 90.3 | 78.6 |
| **Full query, long side 224** | **72.09** | **48.71** | **89.89** | **77.93** |
| Full query, 336 | 73.47 | 51.32 | 90.98 | 79.94 |
| Full query, 448 | 73.36 | 51.36 | 91.18 | 80.51 |
| Full query, 518 | 73.19 | 50.40 | 91.38 | 80.89 |
| Full query, 896 | 71.95 | 46.83 | 90.96 | 80.18 |
| Box-cropped query, 224 (standard) | 63.86 | 40.66 | 88.11 | 74.65 |
| Box-cropped query, 448 | 67.15 | 48.25 | 88.91 | 76.19 |
| Box-cropped query, short side + center crop 224 | 58.20 | 33.29 | 86.85 | 73.57 |

**Gate 1 passes:** full queries at 224 come within 0.4–0.8 mAP of the published numbers on all four columns. That the paper used uncropped queries is *inferred* from this match, not stated in the paper. GeM on patch tokens is far below CLS (for example ROxford M 39.2 vs 63.9), so it was dropped.

Baselines with registers (the model used for everything below):

| Run | Query | Ox M | Ox H | Par M | Par H |
|---|---|---|---|---|---|
| E0: global 224 | full | 74.27 | 51.25 | 90.57 | 79.13 |
| **E1: global 448** | full | **75.45** | **53.37** | **91.80** | **81.38** |
| E0: global 224 | box | 67.28 | 46.02 | 88.38 | 75.25 |
| E1: global 448 | box | 67.63 | 44.96 | 89.32 | 76.77 |

**Warning sign:** cropping the query to its *ground-truth* box lowers ROxford mAP by 6–8 points. The global descriptor uses the context around the object.

### 4.3 E2/E3/E2b: one crop per image (`runs/E2`, `runs/E2_global448`)

**Pipeline:**
1. Localize with last-layer CLS attention: τ = 0.6, K = 1, 10% margin, minimum side 64 px.
2. Crop from the original image and pad to a square with the ImageNet mean colour.
3. Resize to 224 and embed with the CLS descriptor.
4. Score each database image by the maximum cosine over crop pairs.

**Variants:**
- **E2:** localization at 224, query localized the same way.
- **E3:** localization at 448.
- **E2b:** the query cropped by its given box, matched against the same database crops.
- **Fusion:** `0.5·global + 0.5·crop`, with α = 0.5 fixed by the roadmap and not tuned.

| Run (fused with global 448) | Ox M | Ox H | Par M | Par H |
|---|---|---|---|---|
| E1: global 448, full query | 75.45 | 53.37 | 91.80 | 81.38 |
| E2: crop only (localization 224) | 72.15 | 50.25 | 89.59 | 77.91 |
| E3: crop only (localization 448) | 68.00 | 45.89 | 90.06 | 79.00 |
| E2 + global 448 | 75.50 | 53.02 | 91.34 | 80.62 |
| E3 + global 448 | 74.37 | 51.88 | 91.67 | 81.32 |
| Global 448, box query | 67.63 | 44.96 | 89.32 | 76.77 |
| E2b: crop, box query (localization 224) | 68.79 | 46.57 | 88.06 | 75.15 |
| E2b + global 448 (localization 224) | 69.94 | 48.71 | 89.13 | 76.53 |
| E2b + global 448 (localization 448) | 69.48 | 49.06 | 89.52 | 77.18 |

Fusing E2 with the *224* global (75.23 / 53.38) looked like a +2 gain over E0, but E1 at 448 gives the same: it is a resolution effect.

**Localization quality on landmarks** (our box vs the 140 given query boxes):

| Localization size | ROxford mean IoU | IoU ≥ 0.5 | RParis mean IoU | IoU ≥ 0.5 | Fallback |
|---|---|---|---|---|---|
| 224 | 0.42 | 37% | 0.54 | 56% | < 1% |
| 448 | 0.43 | 44% | 0.56 | 60% | < 0.1% |

**Controls for K = 1** (`runs/E2_control_*`). Same pipeline, different box source; E2b + global 448, Hard mAP:

| Box source | Ox H | Par H |
|---|---|---|
| global 448 only | 44.96 | 76.77 |
| fixed central 60% box | 45.78 | 76.14 |
| attention box size, moved to the centre | 46.78 | 76.12 |
| **whole image, padded to a square** | **48.09** | 76.27 |
| attention box, localization 224 | 48.71 | 76.53 |
| attention box, localization 448 | 49.06 | 77.18 |

Paired bootstrap, attention vs the whole-image control: ROxford +0.62 [−0.39, +1.56] (localization 224) and +0.96 [−1.41, +3.91] (448). RParis +0.26 [−0.33, +0.90] and +0.91 [+0.12, +1.78].

**Conclusion:** with K = 1, the gain is a **multi-view ensemble**. Any second view of the whole image gives most of it, and localization adds at most about +1, within noise.

### 4.4 E6: three crops per image, with controls (`runs/E6`, `runs/E6_control_*`, `runs/E6/bootstrap.txt`)

K = 3 attention crops per image (2.4 on average at localization 224), scored by max similarity. Standard protocol (query cropped by its box), Hard mAP, with the bootstrap difference against the strongest baselines without localization:

| Run | Ox H | vs whole image + global [95% CI] | vs centre-matched K = 3 + global [95% CI] |
|---|---|---|---|
| global 448 | 44.96 | −3.1 | −3.2 |
| whole image + global (K = 1 control) | 48.09 | — | 0.0 |
| centre-matched K = 3 + global | 48.12 | 0.0 | — |
| random-matched K = 3 | 36.54 | −11.6 [−18.6, −5.3] | −11.6 |
| **E6: attention K = 3, localization 224** | **52.88** | **+4.78 [+2.77, +6.84]** | **+4.76 [+1.04, +7.87]** |
| E6: attention K = 3, localization 448 | 49.53 | +1.44 [−1.59, +4.36] | +1.41 [−2.50, +4.85] |
| E7: E6 at 224 + global | 51.58 | +3.49 [+2.27, +4.82] | +3.46 [+0.42, +5.93] |
| E7: E6 at 448 + global | 50.43 | +2.34 [+0.86, +3.83] | +2.32 [−0.74, +4.64] |

On RParis Hard, E6 at 448 reaches 78.38: +2.11 [+0.51, +3.92] over the whole-image control and +1.00 [−0.44, +2.51] over centre-matched K = 3.

With three crops, the position of the boxes matters (52.9 attention vs 40.9 centre-matched vs 36.5 random). This is the first localization effect that survives controls. With full queries, E6 + global gives no gain (ROxford H 53.43 vs 53.37).

### 4.5 Asymmetric: whole query, cropped database (`runs/asymmetric`)

Hypothesis: the full-query setting fails because *query-side* localization is poor (IoU 0.42 on ROxford). Test: no query localization at all. The query is the whole image, square-padded at 224. Each database image offers its K = 3 attention crops plus its whole view, and the score is the maximum similarity.

| Database crops (+ whole view) | Ox H | vs E1 [95% CI] | Par H | vs E1 [95% CI] |
|---|---|---|---|---|
| E1: global 448 | 53.37 | — | 81.38 | — |
| **attention K = 3 (localization 224 for Ox, 448 for Par)** | **56.91** | **+3.54 [+1.11, +5.76]** | **83.64** | **+2.25 [+0.91, +3.74]** |
| centre-matched K = 3 (localization 224) | 53.47 | +0.10 [−2.26, +2.27] | 82.89 | +1.51 [+0.27, +2.98] |
| random-matched K = 3 (localization 224) | 52.87 | −0.50 [−2.87, +1.46] | 81.44 | +0.06 [−0.98, +1.30] |
| centre-matched K = 3 (localization 448) | 53.22 | −0.15 | 82.13 | +0.75 [−0.18, +1.78] |
| random-matched K = 3 (localization 448) | 53.08 | −0.29 | 80.95 | −0.43 [−1.32, +0.52] |

The full-query gain appears once the query is left whole, and position-free database crops do not produce it on ROxford. With one fixed localization size for both datasets, the second dataset's CI includes zero: localization 224 gives RParis +0.98 [−0.11, +2.17], and localization 448 gives ROxford +2.20 [−0.44, +4.66].

### 4.6 Better query-side localization (`runs/query_select`)

About 31 candidate crops per query:
- the whole image;
- the attention top-3 at 224 and at 448;
- up to 30 k-means/Hopfield proposals from blocks 8 and 11 at 448, covering 5–90% of the area, de-duplicated with NMS 0.7.

Selection rules that use no labels:
- **attention:** the top attention box.
- **database peak:** mean of the top-10 database scores minus the mean of ranks 100–1000.
- **database top-1:** highest single database score.
- **oracle:** highest IoU with the given box (uses labels; an upper bound).

ROxford, database = attention K = 3 at 224 + whole view; Q0 is the whole query alone (56.91):

| Selection | Query-box IoU | IoU ≥ 0.5 | Crop alone, H | Whole + crop, H | Whole + crop vs Q0 [95% CI] |
|---|---|---|---|---|---|
| attention | 0.421 | 37% | 53.84 | 55.88 | −1.03 [−2.54, +0.28] |
| database peak | 0.329 | 33% | 51.42 | 57.19 | +0.29 [−0.55, +1.27] |
| database top-1 | 0.426 | 41% | 54.94 | 57.01 | +0.10 [−1.36, +1.47] |
| **oracle** | 0.652 | 79% | 52.07 | 58.01 | +1.10 [−0.06, +2.42] |

On RParis all rules are within ±0.3 of Q0 when combined with the whole query, and the oracle is +0.1 to +0.3. 79% (ROxford) and 100% (RParis) of queries have a candidate with IoU ≥ 0.5, but even the oracle crop *alone* is 5 points below the whole query on ROxford.

**Conclusion:** a landmark query frames its own object and needs its context. **Keep the query whole and localize the database.**

### 4.7 Database side: more attention crops vs k-means/Hopfield proposals (`runs/db_eval`)

Database proposals: blocks 8 and 11 at 448; k-means with k ∈ {4, 8} and Hopfield with perplexity ∈ {10, 30}; 5–90% of the area; NMS 0.7. They are ranked by last-layer attention mass and the top k are kept. Controls use the same boxes at random positions (seeded). Generation took about 7–8 minutes per dataset on one GPU.

Full-query protocol (whole query; database = crop set + whole view):

| Database crop set | Crops/image | Ox M | Ox H | vs E1 | Par M | Par H | vs E1 |
|---|---|---|---|---|---|---|---|
| E1: global 448 | — | 75.45 | 53.37 | — | 91.80 | 81.38 | — |
| whole view only | 0 | 73.34 | 50.78 | −2.6 | 90.65 | 79.29 | −2.1 |
| attention K = 3, localization 224 | 2.4 | 76.73 | 56.91 | +3.5 | 91.96 | 82.36 | +1.0 |
| attention K = 5, localization 224 | 3.0 | 76.67 | 56.79 | +3.4 | 91.97 | 82.38 | +1.0 |
| attention K = 5, localization 448 | 2.3 | 76.29 | 56.11 | +2.7 | 92.49 | 83.76 | +2.4 |
| proposals top-3 | 3 | 76.27 | 55.62 | +2.3 | 92.36 | 83.21 | +1.8 |
| proposals top-5 | 5 | 77.77 | 58.48 | +5.1 | 92.89 | 84.48 | +3.1 |
| proposals top-10 | 10 | 78.76 | 60.61 | +7.2 | 93.66 | 85.77 | +4.4 |
| attention K = 3 + proposals top-5 | 7.4 | 78.59 | 60.25 | +6.9 | 93.11 | 85.05 | +3.7 |

Mass-threshold attention gives only about 3 components per image, so K = 5 adds almost nothing over K = 3. At equal count (3), proposals and attention are close. Proposals keep supplying distinct regions as K grows.

### 4.8 How many proposals? K sweep (`runs/db_eval_k`)

Proposals regenerated with up to 30 per image. After filtering, the average is 25.7 per image (at least 10, median 26–27). The first 10 per image were checked to be identical to the earlier set.

Full-query protocol, Hard mAP; "vs random" is a paired bootstrap against the same number and sizes of boxes at random positions:

| K | Crops/image | Ox M | Ox H | vs E1 | vs random [95% CI] | Par M | Par H | vs E1 | vs random [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 10.0 | 78.76 | 60.61 | +7.24 | +3.31 [+1.87, +4.83] | 93.66 | 85.77 | +4.39 | +2.19 [+1.49, +2.98] |
| **15** | 15.0 | **79.38** | **61.93** | **+8.56** | **+3.41 [+1.96, +4.94]** | **93.97** | **86.27** | **+4.89** | **+1.94 [+1.24, +2.75]** |
| 20 | 19.8 | 79.65 | 62.40 | +9.03 | +3.05 [+1.55, +4.53] | 94.02 | 86.33 | +4.95 | +1.55 [+0.94, +2.22] |
| 30 | 25.7 | 79.69 | 62.53 | +9.16 | +1.03 [−0.25, +2.25] | 94.12 | 86.49 | +5.11 | +1.24 [+0.75, +1.82] |

Random-position controls alone (full query, Hard): ROxford 57.30 / 58.52 / 59.35 / 61.50 and RParis 83.58 / 84.33 / 84.78 / 85.24 at K = 10 / 15 / 20 / 30.

Standard protocol (query cropped by its box), Hard mAP:

| K | Ox H | vs E1 (44.96) | vs random [95% CI] | Par H | vs E1 (76.77) | vs random [95% CI] |
|---|---|---|---|---|---|---|
| 10 | 59.18 | +14.22 | +4.21 [+2.89, +5.49] | 82.47 | +5.70 | +3.33 [+2.35, +4.44] |
| **15** | **60.92** | **+15.96** | **+3.99 [+2.36, +5.63]** | **83.44** | **+6.67** | **+3.08 [+2.12, +4.14]** |
| 20 | 61.37 | +16.41 | +3.15 [+1.67, +4.59] | 83.52 | +6.76 | +2.57 [+1.80, +3.43] |
| 30 | 61.67 | +16.71 | +0.64 [−0.99, +2.24] | 83.64 | +6.87 | +1.89 [+1.23, +2.64] |

Findings:
- **The total gain levels off.** ROxford H gains +1.3 from K = 10 to 15, +0.5 from 15 to 20, and +0.1 from 20 to 30.
- **The share due to localization peaks at K = 10–15** and fades by K = 30, where 25+ boxes cover the image densely and position stops mattering.
- **Dense random multi-crop is itself a strong baseline** (ROxford H 61.5 at K = 30). Its standard form, R-MAC-style grid regions, must be in the paper's baselines.
- **Recommended setting: K = 15.** It gets about 93% of the K = 30 gain on ROxford, the largest margin over random boxes, and 16 vectors per database image.

---

## 5. Experiment matrix (roadmap Experiment 1, filled)

Register model. **Full** = whole query (the setting of the published DINOv2 numbers); **Box** = query cropped by its given box (standard revisitop protocol).

| Run | Database | Query | Ox M | Ox H | Par M | Par H |
|---|---|---|---|---|---|---|
| E0 | global 224 | full | 74.27 | 51.25 | 90.57 | 79.13 |
| E1 | global 448 | full | 75.45 | 53.37 | 91.80 | 81.38 |
| E2 | attention crop K = 1, localization 224, square 224 | full, localized | 72.15 | 50.25 | 89.59 | 77.91 |
| E3 | as E2, localization 448 | full, localized | 68.00 | 45.89 | 90.06 | 79.00 |
| E4 | aspect-preserving crop resize | — | not run | | | |
| E5 | PCA masks | — | not run (PCA was 23% fallback on COCO) | | | |
| E6 | attention K = 3, max-sim, localization 224 | full, localized | 69.78 | 47.02 | 88.95 | 76.62 |
| E7 | E6 + global 448 (α = 0.5) | full, localized | 75.45 | 53.43 | 91.82 | 81.26 |
| E8 / E2b | attention K = 3, localization 224 | box | 71.67 | 52.88 | 89.51 | 77.64 |
| new: Q0 | attention K = 3 + whole view | full, **not localized** | 76.73 | 56.91 | 91.96 | 82.36 |
| **new: P15** | **proposals top-15 + whole view** | full, **not localized** | **79.38** | **61.93** | **93.97** | **86.27** |
| **new: P15-box** | proposals top-15 + whole view | box | 79.26 | 60.92 | 93.03 | 83.44 |

Roadmap Experiment 2 (multi-layer localization) was partly covered by Sections 3.3–3.7: per-layer CLS attention, per-layer patch features, resolution, Hopfield grouping and ranking. Experiment 3 (verification) is motivated by Section 3.7 but not started.

---

## 6. Caveats and threats to validity

1. **Choices made on test queries.** Localization size (224 vs 448), K, and the proposal source were chosen after looking at ROxford/RParis results. Mitigations:
   - Every variant run is reported, not only the best.
   - Trends are monotone and agree across both datasets and both protocols.
   - α = 0.5, τ = 0.6, margin 10% and minimum side 64 px are roadmap defaults that were never tuned.

   K = 15 should be **frozen now** and confirmed on a held-out benchmark (ILIAS) without re-tuning.
2. **70 queries per dataset.** Differences under about 1–2 mAP are within noise. Use the bootstrap CIs, not point estimates.
3. **Published protocol inferred.** The DINOv2 paper does not state its retrieval protocol. "Full queries at 224" is inferred from a match within 0.8 mAP.
4. **The baseline that matters is dense multi-crop, not E1.** Random-position crops gain +4 to +8 over E1 by themselves. The localization claim rests on the "vs random" columns (+3.4 ROxford, +1.9 RParis at K = 15) and must be repeated against R-MAC-style grid regions.
5. **Storage and distractors.** 16 vectors per image instead of 1. With R1M distractors that is about 16M vectors (about 25 GB in fp16), which needs PQ or a cap. More crops also give distractors more chances to match by chance. This is untested.
6. **COCO proxy limits.** COCO recall rewards finding *every* labeled object, while retrieval needs the query's object. COCO does not label buildings. COCO size bins here use box area, not mask area.
7. **No version control.** The project is not a git repository. Each run folder has its config (`config*.yaml`) and metrics, but no commit hash.

---

## 7. Next steps (recommended order)

1. **Grid-region baseline:** R-MAC-style multi-scale grid boxes at K = 15, run through the same pipeline. P15 must beat it for the localization claim.
2. **ILIAS** with the frozen configuration (P15, whole query): a held-out test, and the benchmark where database cropping should matter most.
3. **+R1M distractors** with PQ-compressed crop vectors: tests robustness to chance matches and the memory cost.
4. **Learned ranker** for proposals (roadmap Experiment 3, group E): logistic regression or LightGBM on per-layer contrast, attention mass, size, layer and source, trained on COCO train. The oracle shows up to about 3× recall headroom at K = 3–5, which could reach the same mAP with fewer stored vectors.
5. **ViT-L/14** for final numbers.

---

## 8. Reproducing

```bash
# COCO localization (Part A)
python coco_loc.py                                   # 224/448, all mask rules, margins → runs/coco_loc
python coco_loc.py --sizes 224 448 640 0 --margins 0 --vis 20 --out runs/coco_loc_native
python coco_loc.py --sizes 448 --taus 0.7 --margins 0 --layers 2 3 4 5 6 7 8 9 10 11 --vis 0 --out runs/coco_loc_layers
python probe_layers.py --sizes 224 448 640 0 --out runs/probe_native_vs_resize
python probe_layers.py --sizes 0 640 896 1280 --out runs/probe_native
python hopfield_regions.py --group-k 3 10 30                     # → runs/hopfield_regions
python hopfield_regions.py --group-k 3 10 30 --score-k 30 --rankers unranked attention contrast_own_k30 oracle count_bound --out runs/hopfield_oracle

# Retrieval (Part B); datasets in /home/jahongir/Documents/Projects/Dataset/revisitop
python retrieval.py --weights weights/dinov2_vitb14_pretrain.pth --registers 0 --query full --out runs/E0_protocol --sizes 224 336 448 518 672 896
python retrieval.py --query full --out runs/E0_protocol --sizes 224 448          # register model
python crop_retrieval.py --global-size 448 --out runs/E2_global448                # E2/E3/E2b
python crop_retrieval.py --global-size 448 --boxes full --out runs/E2_control_full
python crop_retrieval.py --K 3 --global-size 448 --out runs/E6
python crop_retrieval.py --K 3 --global-size 448 --boxes random-matched --out runs/E6_control_random-matched
python crop_retrieval.py --K 3 --global-size 448 --boxes center-matched --out runs/E6_control_center-matched
python bootstrap_e6.py
python asymmetric.py --loc 224; python asymmetric.py --loc 448
python query_select.py --db-loc 224
python db_proposals.py                                                             # top-10 proposals + random controls (db_eval.py)
python db_proposals.py --max-props 30                                              # top-30 (db_eval_k.py)
python db_eval.py; python db_eval_k.py
```

Caches: COCO attention maps in `data/coco_loc/`, global descriptors in `data/revisitop/*.npz`, crop embeddings in `data/revisitop/crops/`, proposals in `data/revisitop/proposals/`.
