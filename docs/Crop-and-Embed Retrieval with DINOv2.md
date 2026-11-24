# First Experiment Roadmap: Crop-and-Embed Retrieval with DINOv2

Nov 24, 2026 · @Jahongir Yunusov

## Goal and setup

The first experiment tests one question: does cropping the DINOv2-localized object and re-embedding it beat the plain DINOv2 global descriptor? **Success = at least +1 mAP on ROxford Hard or ILIAS** (Gate 2 in the paper plan). Below that, the localization signal is too weak to build on as it stands.

**Pipeline under test**

1. Image → resize → DINOv2 → object regions from the attention map.
2. Boxes scaled back to the original image → crop → resize → DINOv2 → embedding.
3. Same for database and query images → cosine search → mAP.

**Setup**

- **Model:** DINOv2 ViT-B/14 with registers (`dinov2_vitb14_reg` from `torch.hub`, repo `facebookresearch/dinov2`). Patch size 14; tokens are `[CLS, 4 registers, patches]`. ViT-L/14 later for final numbers.
- **Datasets for the pilot:** ROxford5k and RParis6k (revisited, 70 queries each; 4,993 and 6,322 database images), then ILIAS. Add the R1M distractors only after the pipeline works.
- **Code base:** the `revisitop` evaluation code for Oxford/Paris and the `ilias-vrg/ilias` repository for ILIAS.
- **Environment:** PyTorch, `faiss` (GPU if available), `numpy`, `Pillow`, `scipy` (connected components), `pandas` for metadata. One GPU is enough for the pilot.
- **Fixed seeds and a config file** per run, so every number in the results table maps to one config.

## Step 1: Loading images

Load every image once, at full resolution, and keep it: the crop in step 5 must come from the original, not from the resized copy.

- Open with Pillow and convert to RGB (`Image.open(p).convert("RGB")`); this handles grayscale, CMYK and alpha.
- Apply EXIF orientation (`ImageOps.exif_transpose`) before anything else, or boxes will not line up with what you see.
- Store per image: `image_id`, path, original width `W` and height `H`.
- Normalize later with ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`, as DINOv2 expects.
- Use a `DataLoader` with several workers; return the original PIL image and its id, and do the resizing in the dataset's `__getitem__`.

## Step 2: Resizing for the localization pass

Keep the aspect ratio and make both sides multiples of 14; do not squash to a square. DINOv2 accepts non-square inputs and interpolates its position embeddings.

- **Resolution:** run your original 224 setting as one arm, but expect 448 to localize better. At 224 the patch grid is only 16×16, so a small object covers a few cells; at 448 it is 32×32.
- **Recipe:** scale so the long side equals `S` (224 or 448), then round each side to the nearest multiple of 14: `w' = round(W·s / 14)·14`, `h' = round(H·s / 14)·14`, with `s = S / max(W, H)`.
- **Record the factors** `sx = W / w'` and `sy = H / h'`; step 4 needs them.
- **Interpolation:** bicubic with antialiasing for downscaling.
- Grid size after the backbone: `gh = h' / 14`, `gw = w' / 14`.

```python
s = S / max(W, H)
w2 = max(14, round(W * s / 14) * 14)
h2 = max(14, round(H * s / 14) * 14)
img_r = img.resize((w2, h2), Image.BICUBIC)
sx, sy = W / w2, H / h2
```

## Step 3: Finding object regions

Start with the last-layer CLS attention, averaged over heads; compare one alternative (patch-feature PCA) in the same run. Everything happens on the `gh × gw` patch grid.

**3a. Get the CLS-to-patch attention.** The hub model's attention is fused, so recompute the last block's attention by hand:

```python
x = model.prepare_tokens_with_masks(img_tensor)   # [B, 1+R+P, C]
for blk in model.blocks[:-1]:
    x = blk(x)
last = model.blocks[-1]
y = last.norm1(x)
B, N, C = y.shape
H = last.attn.num_heads
qkv = last.attn.qkv(y).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
q, k = qkv[0] * last.attn.scale, qkv[1]
attn = (q @ k.transpose(-2, -1)).softmax(-1)      # [B, H, N, N]
R = model.num_register_tokens                       # 4 for *_reg models
cls_attn = attn[:, :, 0, 1 + R:]                    # [B, H, P]
maps = cls_attn.reshape(B, H, gh, gw)
```

Check this once against `model.forward_features` by running the last block normally: the output tokens must match.

**3b. Turn the map into a mask.**

- **Method A (attention):** average over heads, normalize to sum 1, and keep the smallest set of patches holding 60% of the attention mass (the DINO paper's visualization rule). Also try 50% and 70%.
- **Method B (PCA):** take the last-layer patch features, run PCA per image, keep patches with a positive first component, and flip the sign so the side with more CLS attention is positive.
- Optional later: per-head masks for the head-agreement signal.

**3c. From mask to boxes.**

1. Label connected components on the grid (`scipy.ndimage.label`, 8-connectivity).
2. Rank components by the attention mass they hold.
3. Keep the top K (K = 1 for the first run, then 3), dropping components under 1% of the grid.
4. Box per component, in patch coordinates: `(r0, c0, r1, c1)`, inclusive.
5. Store a simple score per box: its attention-mass share. This is the seed of the reliability score.

**3d. Fallbacks.** No component left, or the top box covers over 90% of the grid: use the whole image as the region and flag it. Log how often this happens per dataset.

## Step 4: Scaling boxes to the original image

A patch cell `(r, c)` covers pixels `[c·14, (c+1)·14) × [r·14, (r+1)·14)` in the resized image; multiply by the step 2 factors to reach the original.

```python
x0 = c0 * 14 * sx;        y0 = r0 * 14 * sy
x1 = (c1 + 1) * 14 * sx;  y1 = (r1 + 1) * 14 * sy
# margin: grid boxes are coarse, so grow each side by 10% of the box size
mw, mh = 0.10 * (x1 - x0), 0.10 * (y1 - y0)
x0, y0 = max(0, x0 - mw), max(0, y0 - mh)
x1, y1 = min(W, x1 + mw), min(H, y1 + mh)
```

- Try margins of 0%, 10% and 20%; context around the object often helps the embedding.
- Enforce a minimum crop size (e.g. 64 px per side in the original) so tiny boxes do not upsample noise.
- **Check it visually:** draw boxes on 50 random images per dataset before running anything at scale.

## Step 5: Cropping and embedding

Crop from the original image, keep the crop's aspect ratio, and embed it with the same frozen model. Always compute the full-image global embedding too: it is the baseline and the fallback.

- **Crop:** `img.crop((x0, y0, x1, y1))` on the original PIL image.
- **Resize the crop:** your original plan resizes to 224×224, which stretches non-square crops. Compare two arms: (a) pad the crop to a square with the ImageNet mean colour, then resize to 224; (b) long side to 224 or 336, sides rounded to multiples of 14, no padding.
- **Embedding:** take `x_norm_clstoken` from `model.forward_features` as the first descriptor. As a second, GeM (p = 3) over `x_norm_patchtokens`. L2-normalize both.
- **Precision:** run the model in fp16 or bf16; store vectors as fp16.
- **Batching:** crops have different sizes, so group them by output shape, or use the padded square arm for one-shape batches.

```python
with torch.no_grad():
    out = model.forward_features(batch)        # batch: normalized crops
    cls = F.normalize(out["x_norm_clstoken"], dim=-1)
    pt  = out["x_norm_patchtokens"].clamp(min=1e-6)
    gem = F.normalize(pt.pow(3).mean(1).pow(1 / 3), dim=-1)
```

Note that GeM on DINOv2 tokens needs the clamp because tokens can be negative; if GeM does worse than CLS, keep CLS and move on.

## Step 6: Building the database

Store every vector with the image it came from, so you can rank images, not crops.

| File | Content | Shape or columns |
| --- | --- | --- |
| `global.npy` | Full-image embedding per database image | `[N, D]`, fp16 |
| `crops.npy` | All crop embeddings | `[M, D]`, fp16, M ≤ N·K |
| `crops_meta.parquet` | One row per crop | `crop_idx`, `image_id`, box `x0 y0 x1 y1`, score, rank k, fallback flag |
| `images.parquet` | One row per image | `image_id`, path, `W`, `H` |
| `config.yaml` | Every setting that produced the files | model, S, mask rule, K, margin, crop resize arm |

- **Index:** for ROxford and RParis, an exact inner-product search is enough (`faiss.IndexFlatIP`, or a plain matrix product on the GPU). With R1M distractors, use a GPU `IndexFlatIP` or PQ only after exact results are in.
- **Multi-crop scoring:** search the crop vectors, then score each database image by the maximum similarity over its crops. Search deeper than the final list (e.g. top 1000 × K) before aggregating.
- **Whitening:** none in the first run. Later, learn PCA-whitening on a separate set (not the test database) and report both.
- **Cache everything:** the localization pass and the crop embeddings are the slow parts; a new fusion or scoring rule should never need a forward pass.

## Step 7: Query handling and evaluation

Run the query through steps 1 to 5 exactly as a database image, but respect each benchmark's query protocol, or the numbers will not compare with published ones.

**Query protocol per dataset**

- **ROxford / RParis:** the standard protocol crops each query with its given bounding box. Report two settings: (1) standard, query cropped by the given box, with your method only on the database side; (2) full query image with your own localization. Setting 2 is non-standard, so label it clearly; it directly tests your localization.
- **ILIAS:** queries show the object on a clean background and positives show it in clutter, so the database side is where cropping should matter most. Use the queries as given.

**Scoring a query**

1. Global baseline: cosine between query global and database global.
2. Crop method: query crop vector(s) against database crop vectors, max over crop pairs per database image.
3. Fusion (later): `s = α·s_global + (1 − α)·s_crop`, with α in {0.25, 0.5, 0.75}; never tune α on the test queries of the table you report.

**Metrics**

- **ROxford / RParis:** mAP for the Medium and Hard protocols with the `revisitop` ground-truth files and `compute_map` (Medium: easy + hard are positives; Hard: only hard are positives, easy is ignored). Add +R1M once the pipeline is stable.
- **ILIAS:** the repository's evaluation script and metric (the paper reports mAP@1k; confirm in the repo). Start with the 1,000 instances and their positives, then add distractors.
- **Also log:** query time, extra forward passes per image, crops per image, fallback rate.

**Localization check for free:** the 70 ROxford and 70 RParis queries come with boxes, so run your localization on the full query images and report mean IoU with the given boxes. It is the first evidence for the reliability-score idea.

## Experiment matrix

Run the rows top to bottom; each changes one thing from the row above or from E0. Fill the cells as results come in.

| Run | Database side | Query side | ROxf M | ROxf H | RPar M | RPar H | ILIAS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | Global, 224 | Global, 224 |  |  |  |  |  |
| E1 | Global, 448 | Global, 448 |  |  |  |  |  |
| E2 | Crop K=1, loc 224, crop 224 square (your plan) | Same |  |  |  |  |  |
| E3 | Crop K=1, loc 448 | Same |  |  |  |  |  |
| E4 | E3 + aspect-preserving crop resize | Same |  |  |  |  |  |
| E5 | E4 with PCA masks (method B) | Same |  |  |  |  |  |
| E6 | Best of E3–E5, K=3, max-sim | Same |  |  |  |  |  |
| E7 | E6 + fusion with global (α = 0.5) | Same |  |  |  |  |  |
| E8 | Best run | Given query box (standard protocol) |  |  |  |  | n/a |

**Also record per run:** mean IoU of the localized box against the ROxford/RParis query boxes, fallback rate, and query latency.

**Reading the table:** E2 vs. E0 answers the main question; E3 vs. E2 shows how much the 224 grid costs; E1 vs. E0 shows how much of any gain is just resolution.

## Schedule

&#91;embedded content: Pilot schedule · Nov 24 to Dec 27, two gates\]

Gate 1 (Dec 6) checks that E0 reproduces published DINOv2 numbers within about 1 mAP. Gate 2 (Dec 27) asks for at least +1 mAP over E0 on ROxford Hard or ILIAS; these dates match the paper plan's roadmap.

## Experiment 2: Multi-layer localization for small objects

Experiment 2 builds on the baseline above: it keeps the best Experiment 1 run (call it E-best) fixed and changes only how regions are found. The question is whether DINOv2's middle layers recover small objects that the last layer misses. Start it once E-best is known.

**Why the last layer misses small objects**

- The last layer's CLS attention concentrates on the dominant object, so a small one gets little mass and the threshold removes it.
- At low input resolution a small object covers only one or two patches in every layer.

**Region sources to compare**

1. **Mid-layer CLS attention.** CLS attention from layers 6 to 10 (ViT-B, 12 layers), fused by a weighted average or by a union of per-layer masks.
2. **Last layer cleans, mid layers find.** Mark clear background from the last-layer map, then look for objects in mid-layer maps only in what remains.
3. **Patch-feature clustering.** k-means or a normalized cut (as in TokenCut) on mid-layer patch features; each compact cluster that differs from its surroundings is a candidate.
4. **Local contrast.** Score each patch by how unlike its neighbours its mid-layer feature is; high-contrast blobs are candidates even when CLS ignores them.
5. **Per-head maps.** Components taken from each head separately instead of the head average, which washes small objects out.
6. **Higher resolution or tiles.** Localization at 448 or 518, or overlapping tiles for large images, so small objects cover more patches in every layer.

To get attention for several layers, run the blocks one by one and recompute each chosen block's attention the way step 3a does for the last one:

```python
layers = [6, 7, 8, 9, 10, 11]          # 0-based block indices; 11 = last for ViT-B
maps = {}
x = model.prepare_tokens_with_masks(img_tensor)
for i, blk in enumerate(model.blocks):
    if i in layers:
        maps[i] = cls_attention(blk, x, R)   # step 3a's code as a function: [B, H, P]
    x = blk(x)
```

**Keeping the noise down**

- Skip early layers (below about 4 of 12): they respond to edges and texture.
- Keep a candidate only if it appears in at least two layers or heads, is compact, and passes a minimum size.
- Filter artifact tokens (use the register model and check token norms).
- More candidates means more false ones, so use top-K crops with max-similarity matching and the confidence score from the reliability work.

**Evaluation, part 1: localization (no retrieval yet)**

- On COCO val, split ground-truth boxes into COCO's small, medium and large bins.
- For each region source and each single layer, report box recall: the share of ground-truth objects with a candidate box at IoU ≥ 0.5, at K = 1, 3 and 5 candidates per image.
- Also report the mean number of candidates per image, since recall alone rewards proposing everything.
- Expected result: the last layer recalls large objects well and small ones poorly; mid-layer sources recover the small ones. The plot of recall by layer for each size bin is a likely paper figure.

**Evaluation, part 2: retrieval runs**

| Run | Region source (all else as E-best) | ROxf H | RPar H | ILIAS | INSTRE |
| --- | --- | --- | --- | --- | --- |
| M0 | Last layer only (= E-best) |  |  |  |  |
| M1 | Single best mid layer |  |  |  |  |
| M2 | Mid-layer fusion, layers 6 to 10 |  |  |  |  |
| M3 | Last-layer-gated mid layers |  |  |  |  |
| M4 | Patch-feature clustering, mid layer |  |  |  |  |
| M5 | Local contrast, mid layer |  |  |  |  |
| M6 | Per-head components |  |  |  |  |
| M7 | Best of M1–M6 at 518 input or with tiles |  |  |  |  |

Report the mAP gain over M0 grouped by object size. On ILIAS and INSTRE, use the box size of positives where annotations allow; otherwise bin by the size of the localized region.

**Success criterion:** higher small-object recall on COCO and a retrieval gain on small-object queries, with no loss on large objects. If small objects improve but large ones drop, the confidence-weighted fusion should decide per image which source to trust.

## Experiment 3: Verifying that a crop is a real object

Experiment 3 adds a verification step between region finding and embedding: each candidate crop gets a score for being a real object rather than noise or background. It keeps the best Experiment 2 setup (M-best) fixed and only adds this score. Test the approaches below on labeled regions first, then keep the two or three best for retrieval.

**Labels for testing the verifiers**

- Run M-best's region finding on COCO val (and LVIS for more small objects).
- Label each candidate box against ground truth: **object** (IoU ≥ 0.5 with a box), **partial** (0.1 to 0.5), **background** (below 0.1, or overlapping no box).
- Add the ROxford/RParis query boxes as a landmark check, since COCO does not label buildings as objects.

**A. Training-free signals from DINOv2 itself**

1. **Cross-layer agreement:** IoU between the region's mask in the last layer and in 2 to 3 mid layers.
2. **Head agreement:** IoU between the masks from individual attention heads.
3. **Inside vs. outside contrast:** mean cosine similarity of patch features inside the mask minus their similarity to patches outside it.
4. **Inside coherence:** mean pairwise similarity of patches inside the mask (high for one object, low for a mix).
5. **Compactness:** number of connected pieces, area ratio (e.g. 1% to 60% of the image), border contact.
6. **Artifact check:** norm of the tokens at the attention peak; very high norm means an artifact token.
7. **Peakedness:** entropy of the attention inside the region.
8. **Stability under augmentation:** mask IoU under a horizontal flip and a small rescale.
9. **Crop–global consistency:** similarity between the crop embedding and the full-image embedding (a soft signal: a real small object can also be far from the global one).
10. **Re-embedding stability:** similarity between the crop's embedding and the embedding of the same crop with a slightly larger or smaller margin. Real objects change little.
11. **Re-localization:** run the localization again on the crop; a real object fills the new mask, background gives a scattered one.

**B. Energy and memory signals**

12. **Hopfield energy against the image's own patches:** pooled region feature as the query, patch tokens as memory, energy = −logsumexp(β·similarities). Tune β on a validation split.
13. **Hopfield convergence:** number of update steps until the retrieved pattern stops changing, and the entropy of the final attention.
14. **Energy against a background memory:** a bank of features from known background patches (sky, road, wall, floor, from COCO-Stuff); low energy against it means the region looks like background.

**C. Signals from the database (retrieval-aware)**

15. **Hubness:** mean similarity of the crop to its k nearest database crops. Very high means generic background or texture that matches everything.
16. **Reciprocal neighbours:** whether the crop's nearest neighbours also have it among their nearest neighbours; real instances tend to have reciprocal matches.
17. **Neighbour consistency:** whether the crop's nearest neighbours come from the same few images or are spread thin across many.

**D. External models (zero-shot, no training by you)**

18. **SAM:** prompt SAM with the box; use its predicted IoU and stability score, and the overlap between SAM's mask and your mask.
19. **Open-vocabulary detectors:** OWLv2 or Grounding DINO objectness score for the box, or the best IoU between your box and their class-agnostic proposals.
20. **Unsupervised detectors:** CutLER or another class-agnostic detector trained without labels; best IoU with its boxes.
21. **CLIP or SigLIP text prompts:** similarity of the crop to “a photo of an object” vs. “a photo of background”, “sky”, “a wall”, “a texture”.
22. **Saliency models:** mean saliency inside the box from a salient-object detector.

**E. Learned verifiers (small, on frozen features)**

23. **Logistic regression on signals 1 to 17:** a few hundred labeled regions are enough; easy to interpret.
24. **Small MLP on the crop's DINOv2 embedding plus the signals,** trained on COCO labels; keep it class-agnostic.
25. **Gradient-boosted trees** (e.g. LightGBM) on the same inputs, as a strong non-neural reference.
26. **Learned objectness from the patch grid:** a tiny head on the region's patch features predicting IoU with ground truth (regression instead of a yes/no class).

Groups D and E are baselines and upper references; groups A to C are where the paper's training-free claim lives.

**Evaluation, part 1: does the score separate the classes?**

- AUROC for object vs. background and for object vs. partial, per signal and for the combinations.
- Spearman correlation between the score and the true IoU, and a plot of score vs. true IoU.
- Results split by object size (small, medium, large) and on the landmark check set.
- Cost per region (milliseconds, extra forward passes).

**How the score is used in retrieval**

- **Filter:** drop crops below a threshold; fall back to the global embedding when none pass.
- **Weight:** fused descriptor d = normalize(g + c·r), with c the verification score.
- **Rank:** keep the top-K crops by score instead of by attention mass.

**Evaluation, part 2: retrieval runs**

| Run | Verification (all else as M-best) | Use | ROxf H | RPar H | ILIAS | INSTRE |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | None (= M-best) | — |  |  |  |  |
| V1 | Best single training-free signal | Filter |  |  |  |  |
| V2 | Combination of group A signals | Weight |  |  |  |  |
| V3 | V2 + Hopfield energy (group B) | Weight |  |  |  |  |
| V4 | V3 + database signals (group C) | Weight |  |  |  |  |
| V5 | SAM or OWLv2 score (group D) | Weight |  |  |  |  |
| V6 | Learned MLP (group E) | Weight |  |  |  |  |
| V7 | Oracle: ground-truth IoU where available | Weight |  |  |  |  |

V7 is an upper bound: it shows how much retrieval could gain from a perfect verifier. A training-free verifier that closes most of the gap between V0 and V7 is a strong result.

**Success criterion:** a training-free score with AUROC of about 0.8 or more on object vs. background, and a retrieval gain over V0 with no loss on landmarks.

## Pitfalls and sanity checks

- **Reproduce E0 first.** If the DINOv2 global mAP on ROxford/RParis is far from published DINOv2 numbers, fix the evaluation before testing anything else.
- **Register tokens.** Slice patch tokens after the CLS and register tokens (`1 + R`), or the map is shifted by four cells.
- **Grid vs. image orientation.** `reshape(gh, gw)` must match `h' / 14` rows and `w' / 14` columns; a swapped shape gives plausible-looking but wrong boxes.
- **EXIF rotation.** Apply it in step 1, or boxes land on the wrong part of rotated photos.
- **Query protocol.** Do not mix the cropped-query and full-query settings in one table.
- **No tuning on test queries.** Pick thresholds, margins and α on a held-out split (e.g. a few GLDv2 queries) or report all settings.
- **Look at failures.** Save the 20 queries with the largest drop vs. E0 and view their boxes; that tells you what the reliability score must catch.
- **Version everything.** Each run writes its config, git commit and metric file to its own folder.
