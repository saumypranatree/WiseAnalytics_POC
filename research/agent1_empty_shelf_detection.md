# Agent 1: Empty-shelf (out-of-stock gap) detection with gap size in cm

Research date: 2026-10-15. Research only; no project files were changed.
Rules followed: every factual claim has the URL it came from. "Unverified" marks anything I could not confirm from a page I opened. Text marked **[reasoning]** is my own engineering judgement, not a sourced fact.

Roboflow Universe dataset pages returned HTTP 403 (a Cloudflare challenge) to both WebFetch and curl. Their details below come only from search-result snippets or from Roboflow's own blog, and are marked that way.

---

## 1. Summary table of approaches

| # | Approach | How it works | Needs training? | Data available | License concerns | Fits our pipeline? | Main risks |
|---|---|---|---|---|---|---|---|
| a | Direct 2D gap detector | Detector trained on a class such as "empty space" / "OOS", optionally also "front" (products pushed back) | Yes (fine-tune) | Small public sets: Roboflow (Policenta about 1,050 images; others 40-333 images). Academic sets (511 and 1,000 images) are not public | Detector: RF-DETR N/S/M/L are Apache-2.0; Ultralytics YOLO is AGPL-3.0 (commercial use needs an Enterprise licence). Roboflow dataset licences per page, mostly unverified | Very good: replace or add to the Grounding DINO step on frames or renders, then lift boxes to 3D exactly as now | Domain shift to Indian stores; small datasets; defining "gap" consistently |
| b | Dense product detection, then gaps between products | Detect every pack (SKU-110K style), group the boxes into shelf rows, and treat horizontal gaps wider than about one facing as empty | Off the shelf (SKU-110K weights), or fine-tune | SKU-110K: 11,762 images, about 1.73M boxes. CC0 Humans-in-the-Loop set: 45 images, 11,743 boxes | **SKU-110K is academic/non-commercial only**, and so are weights trained on it (legal check needed) | Good: product boxes lift to 3D. Gaps are computed per row in 3D, so they come out directly in metres | Missed detections look like false gaps. Cannot see "front"/pushed-back OOS. One published comparison found this weaker than direct detection |
| c | Shelf-edge / price-tag detection to define rows and slots | Detect the shelf-edge strips (yellow price strips) and the tags. Each strip is a row baseline and each tag is a slot anchor | Yes (small detector or segmenter), or try open-vocab | CC0 HiL set (Price class). Roboflow price-tag and shelf sets (unverified) | Same as (a) | Very good as a structural backbone, and the strip also provides scale | Tags do not always align with facings; occlusion |
| d | Open-vocab "empty shelf" prompt (Grounding DINO/OWLv2) | Text prompt such as "empty shelf space" | No | n/a | Grounding DINO code and HF weights Apache-2.0; SAM/SAM2 Apache-2.0 | Already in the pipeline, so trivial to try | **No published evaluation found.** Benchmarks show OVD models fail on fine-grained and relational descriptions. "Emptiness" is the absence of an object, which is a poor match for object-centric detectors **[reasoning]** |
| e | 3D/geometric (depth or mesh) | Fit the shelf front plane and each shelf board. A gap is where the surface lies more than X cm behind the front plane inside a row | No (rules), or light tuning | Our own scans | Depends on the reconstruction tool | Native to the mesh pipeline. Gives cm directly once scale is fixed | Dark backs and reflective bottles break photogrammetry. A thin mesh can fill holes falsely. Shallow products |
| f | Planogram comparison | Align detected products or slots to the expected layout (for example by sequence alignment) | Needs product recognition | Needs the client's planogram | n/a | Later phase | Needs SKU-level recognition and planogram data we do not have |

---

## 2. Details by approach

### (a) Direct 2D gap / OOS detection

- **Šikić et al., Sensors 2024, "Enhanced Out-of-Stock Detection in Retail Shelf Images Based on Deep Learning"** (https://pmc.ncbi.nlm.nih.gov/articles/PMC10819825/).
  - Models: YOLOv5s6, YOLOv7-tiny and EfficientDet-D3, starting from COCO weights, with 1280x1280 input.
  - Pre-training on synthetic images made by copying OOS boxes over neighbouring products.
  - Post-processing that discards detections whose aspect ratio falls outside the per-class/per-section range seen in training.
  - Dataset: 511 images from 4 Croatian retailers, 1,210 OOS instances, in two classes: "normal" (fully empty, 730) and **"front" (products remain at the back, 480)**.
  - **Not public**: available from the authors only with Cloudonia d.o.o.'s permission.
  - Reported (YOLOv5 + front class, 5-fold): normal AP 86.3%, front AP 83.7%, mAP 85.0%. Product-detection-based baseline (Chen et al.): normal AP 56.9-63.8.
  - Failure modes noted: missed empties; false positives from products hanging at shelf ends; large top-shelf gaps only partly localized; visible background through open shelf backs; visible gaps between products (removed by post-processing).
  - The authors list depth estimation as future work.
- **OOS-DSD, Šikić & Lončarić, arXiv 2510.16508** (https://arxiv.org/abs/2510.16508, https://arxiv.org/html/2510.16508).
  - YOLOv8 with extra branches for OOS detection, product segmentation and depth. The depth branch is trained on Depth Anything V2 pseudo-labels, with a depth-normalization step.
  - Reported gains: +1.8% mAP over the previous state of the art, +3.7% from auxiliary learning, +4.2% from depth normalization. Absolute mAP was not readable on the HTML page (unverified).
  - Data: 171 beverage images from the same in-house dataset. The authors state "there are no publicly available OOS detection image datasets with bounding box annotations".
  - No code link. Paper licence CC BY-NC-SA 4.0 (this covers the paper, not a model).
- **Jha et al., arXiv 2205.13060, "Designing an Efficient End-to-end ML Pipeline for Real-time Empty-shelf Detection"** (https://arxiv.org/abs/2205.13060, https://arxiv.org/html/2205.13060).
  - 1,000 annotated images (800/100/100) from US Target stores (paper-towel and tissue aisles) taken with an iPhone 12 Pro Max.
  - Single class (empty location). Annotation rules: label the front face of the empty 3D box; only fully empty locations count; a continuous gap is one box; the gap must be at least half the width of the neighbouring product. These rules are useful as a template for our own labels.
  - Abstract reports mean average F1 of 68.5%. The text names YOLOv5n6 as best with mAP 66.9 / mAR 76.3 / mAF 71.3, which is internally inconsistent with the abstract.
  - The learning curve had not saturated at 800 images.
  - Dataset not stated as public.
- **Roboflow Universe sets.** All pages returned 403, so details come from snippets or blogs.
  - Policenta "Empty shelf detection": 1,050 images, one class "Empty-space" (Roboflow blog, https://roboflow.com/blog/retail-store-object-detection). About 1.1k images on the profile (search snippet, https://universe.roboflow.com/policenta). **Licence unverified.**
  - fyp-ormnr "Empty Shelf Detector": 333 images, class "100% O.O.S" (snippet, https://universe.roboflow.com/fyp-ormnr/empty-shelf-detector). Licence unverified.
  - Roboflow's on-shelf-monitoring tutorial forks a "Supermarket Empty Shelf Detector" set (class "100- O-O-S"), trains RF-DETR Small, and reports **no accuracy numbers** (https://roboflow.com/blog/automate-on-shelf-monitoring).
  - yolo-x2kai (44 images, classes Empty and Bounding-Box): https://universe.roboflow.com/yolo-x2kai/empty-shelf-detection-cbhhp
  - elis-workspace (40 images, classes empty and front): https://universe.roboflow.com/elis-workspace-sucvp/empty-shelf-data
  - hostview, test-py0xd, pro-2kotu and art-ssd sets: snippets say CC BY 4.0, sizes unknown. https://universe.roboflow.com/hostview/empty-shelf-detection-le2j0, https://universe.roboflow.com/test-py0xd/empty-shelves-fnh4e/dataset/, https://universe.roboflow.com/pro-2kotu/shelf_empty
  - All of these need manual license and quality checks (unverified).

### (b) Product detection, then gaps between products

- **SKU-110K** (Goldman et al., CVPR 2019; https://arxiv.org/abs/1904.00853; repo https://github.com/eg4000/SKU110K_CVPR19).
  - 11,762 images, average 147.4 objects per image. Train split: 8,233 images and 1,210,431 boxes; val: 588 images; test: 2,941 images and 432,312 boxes (from the paper PDF).
  - Labels are product boxes only, with no SKU classes.
  - Reported (their method): AP .492, AP.75 .556, AR300 .554.
  - **Licence: "solely for academic and non-commercial purposes"** (repo README). The repo code licence is not stated. A pretrained model is on Google Drive (repo).
  - Ultralytics' YAML for SKU-110K is AGPL-3.0, but that covers their config file, not the data (https://docs.ultralytics.com/datasets/detect/sku-110k).
  - **Flag:** any checkpoint trained on SKU-110K likely inherits the non-commercial restriction. This is a legal question, unverified.
- **Humans in the Loop Supermarket Shelves**: 45 images, 11,743 boxes, classes Product and Price, **CC0 1.0** (commercial OK) (https://humansintheloop.org/resources/datasets/supermarket-shelves-dataset/).
- Other results cited in 2205.13060: Cascade R-CNN on SKU-110K at 58.7% mAP (Rong et al.). Cited, not independently opened.
- Weaknesses: a missed product becomes a false gap, and this approach cannot detect "front"/pushed-back OOS (stated in arXiv 2510.16508). Šikić 2024 found a product-detection baseline much weaker than direct detection (normal AP 56.9-63.8 vs 86.3).

### (c) Shelf-edge / price-tag structure

- No dedicated public shelf-edge dataset was found. The CC0 HiL set has a Price class (link above).
- Roboflow sets:
  - "Shelf Detection 2" (Marketvision): about 16,940 images, CC BY 4.0, per snippet; classes unverified. https://universe.roboflow.com/marketvision-co-ltd/shelf-detection-2
  - hostview "price tag detection": licence unverified. https://universe.roboflow.com/hostview/price-tag-detection-kjttf
- Many patents exist (for example US11200677B2 shelf-edge detection, https://patents.google.com/patent/US11200677). These are relevant for freedom-to-operate review, not as data.
- **[reasoning]** On our footage, the continuous yellow strips look like an easy target for colour and line detection or a small segmenter. They define the row baselines, the shelf front plane in 3D, and a scale reference.

### (d) Open-vocabulary "empty shelf" prompts

- I found **no paper or benchmark that evaluates Grounding DINO / OWL-ViT / OWLv2 on empty-shelf prompts** (searches documented in the sources list).
- OVDEval (arXiv 2308.13177, https://arxiv.org/abs/2308.13177) reports that state-of-the-art OVD models "all fail on the new tasks except for simple object types" on attribute, position and relation descriptions, and that standard AP can be deceptive on such tasks.
- Grounding DINO: code Apache-2.0; checkpoints GroundingDINO-T (COCO zero-shot 48.4 AP) and B (https://github.com/IDEA-Research/GroundingDINO). Hugging Face weights IDEA-Research/grounding-dino-tiny and -base are tagged apache-2.0 (HF API).
- **[reasoning]** Worth a 1-hour baseline on our frames because it costs nothing. Do not expect production quality. A better use of open-vocab is detecting *products* and *price tags*, then reasoning about gaps geometrically.

### (e) 3D / geometric

- Milella et al.:
  - "3D vision-based shelf monitoring system for intelligent retail" (2021) compares a learned reference shelf model with the current depth data from a consumer depth sensor and needs no product priors.
  - IEEE Access 2020 (point-cloud reconstruction, surface fitting, occupancy grids) reports a maximum average discrepancy of about 5.0% in on-shelf availability for fresh products.
  - Demonstrator: Intel RealSense D435.
  - Sources: https://doaj.org/article/302301b555c1481b8a0ef957710e06f5, https://iris.cnr.it/handle/20.500.14243/446128. Seen via search summary; details only partly verified.
- Monocular depth options (licences checked):
  - Depth Anything V2: Small 24.8M params Apache-2.0; **Base/Large CC-BY-NC-4.0 (non-commercial)** (https://github.com/DepthAnything/Depth-Anything-V2). Metric fine-tunes exist for S/B/L, trained on Hypersim (indoor) and VKITTI (outdoor) (https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md). HF tag for Metric-Hypersim-Small is apache-2.0; the B/L metric weights presumably follow the NC licence (unverified).
  - Metric3D v1/v2: code BSD-2-Clause; outputs metric depth and normals; accuracy depends on correct focal length (https://github.com/YvanYin/Metric3D).
  - MoGe-2: code MIT; HF weights tagged MIT; metric scale; ViT-S 35M to ViT-L 326M (https://github.com/microsoft/MoGe).
  - Apple Depth Pro: metric depth plus focal-length estimate without intrinsics. The LICENSE is Apple's permissive sample-code licence (use/modify/redistribute) with no patent grant (https://github.com/apple/ml-depth-pro, LICENSE file). Have legal confirm commercial use.
  - No source I opened reported accuracy for any of these on retail shelves, so no numbers are given here.
- **[reasoning]** Geometric rule on our mesh, per shelf row:
  1. Fit the front plane from the price-strip line.
  2. Fit the shelf board plane and the back panel.
  3. For each column bin along the row, measure the depth of the first surface behind the front plane.
  4. Treat bins whose depth exceeds about 0.5-0.7 of the shelf depth, or which hit the back panel, as an empty candidate.
  5. Count consecutive bins as one gap; width = run length and height = board-to-next-board distance.

  This catches "front" OOS (product pushed far back) as a depth step, which 2D product detection cannot do.

### (f) Planogram comparison

- arXiv 2212.01004 (https://arxiv.org/abs/2212.01004): object detection with local features and implicit shape models, then compliance checking via a modified Needleman-Wunsch sequence alignment. Datasets and metrics were not on the abstract page.
- MVA 2015 smartphone compliance framework (https://www.mva-org.jp/Proceedings/2015USB/papers/14-39.pdf; search snippet only).
- Requires the client's planogram and SKU recognition. Out of scope for phase 1 **[reasoning]**.

---

## 3. Datasets

| Dataset | Size | Labels | Licence | Link |
|---|---|---|---|---|
| SKU-110K | 11,762 images; about 1.73M boxes; 147.4 per image | Product boxes (single class) | **Academic / non-commercial only** | https://github.com/eg4000/SKU110K_CVPR19 |
| HiL Supermarket Shelves | 45 images, 11,743 boxes | Product, Price | CC0 1.0 | https://humansintheloop.org/resources/datasets/supermarket-shelves-dataset/ |
| Policenta Empty shelf (Roboflow) | 1,050 images | Empty-space | Unverified (page 403) | https://universe.roboflow.com/policenta (via https://roboflow.com/blog/retail-store-object-detection) |
| fyp-ormnr Empty Shelf Detector (Roboflow) | 333 images (snippet) | "100% O.O.S" | Unverified | https://universe.roboflow.com/fyp-ormnr/empty-shelf-detector |
| elis-workspace Empty Shelf Data | 40 images (snippet) | empty, front | Unverified | https://universe.roboflow.com/elis-workspace-sucvp/empty-shelf-data |
| yolo-x2kai Empty shelf | 44 images (snippet) | Empty, Bounding-Box | CC BY 4.0 (snippet) | https://universe.roboflow.com/yolo-x2kai/empty-shelf-detection-cbhhp |
| Marketvision Shelf Detection 2 | about 16,940 images (snippet) | Shelf (classes unverified) | CC BY 4.0 (snippet) | https://universe.roboflow.com/marketvision-co-ltd/shelf-detection-2 |
| Retail Store Indian (Roboflow) | Unknown | Indian brands, e.g. Amul (snippet) | CC BY 4.0 (snippet) | https://universe.roboflow.com/nmims-b6ta9/retail-store-indian/dataset/ |
| Šikić OOS (Croatia) | 511 images, 1,210 OOS | normal, front | Not public (by permission) | https://pmc.ncbi.nlm.nih.gov/articles/PMC10819825/ |
| Jha et al. (Target, US) | 1,000 images | empty | Not stated public | https://arxiv.org/html/2205.13060 |
| RP2K | >500k product images, 2,000 products | Classification | Not checked | https://arxiv.org/pdf/2006.12634 |

No verified public **Indian** shelf or OOS dataset was found.

## 4. Pretrained models / checkpoints

| Model | Use | Size | Licence | Source |
|---|---|---|---|---|
| Grounding DINO T / B | Open-vocab detection | Swin-T / Swin-B (file sizes not listed) | Apache-2.0 (code; HF weights) | https://github.com/IDEA-Research/GroundingDINO |
| SAM (vit_b/l/h) | Segmentation | n/a | Apache-2.0 | https://github.com/facebookresearch/segment-anything |
| SAM 2.1 tiny/small/b+/large | Segmentation (video) | 38.9 / 46 / 80.8 / 224.4M | Apache-2.0 | https://github.com/facebookresearch/sam2 |
| RF-DETR Nano/Small/Medium/Large | Detector to fine-tune for gaps, products and tags | 30.5-33.9M | **Apache-2.0** (XL/2XL/Atto/Femto/Pico are PML 1.0) | https://github.com/roboflow/rf-detr |
| RF-DETR-Seg N to 2XL | Instance segmentation | 33.6-38.6M | Apache-2.0 | same |
| Ultralytics YOLO | Detector | n/a | **AGPL-3.0; commercial/closed use needs Enterprise licence** | https://www.ultralytics.com/license |
| SKU-110K RetinaNet | Product detection | n/a | Data non-commercial; code licence unstated | https://github.com/eg4000/SKU110K_CVPR19 |
| Depth Anything V2 S / B / L | Relative or metric depth | 24.8 / 97.5 / 335.3M | S Apache-2.0; **B/L CC-BY-NC-4.0** | https://github.com/DepthAnything/Depth-Anything-V2 |
| Metric3D v2 | Metric depth and normals | S/L/g | BSD-2 (code) | https://github.com/YvanYin/Metric3D |
| MoGe-2 | Metric geometry | 35-331M | MIT (code; HF weights) | https://github.com/microsoft/MoGe |
| Depth Pro | Metric depth and focal length | n/a | Apple sample-code licence (permissive; no patent grant) | https://github.com/apple/ml-depth-pro |

No source I opened documents VRAM figures. **[reasoning]** All of the Apache/MIT models above at S/B/L size should fit in 8 GB for inference; this was not tested.

## 5. Converting a gap to centimetres

1. **3D reconstruction plus a standard-fixture scale (recommended).** The mesh from the video is up-to-scale. Pick one or more references, compute scale = known size / mesh size, and average them:
   - **Price/label-strip height:** HL Display paper-label datastrips use a **39 mm** label height across several profiles (ANTL, LSFN, LST, SU, HEL), for 19-25 mm shelves (https://www.hl-display.com/catalogue/price-communication/datastrips-for-paper-labels/antl). This is **one vendor, not a standard**, and the Indian yellow strips may differ. Measure one in the store.
   - **Bay width:** Indian supplier listings commonly give **3 ft (about 900 mm) or 4 ft (about 1200 mm)** widths and 5-9 ft heights (https://dir.indiamart.com/bengaluru/supermarket-display-rack.html). A Surat gondola is listed at 3 ft length, 1.5 ft depth, 6 ft height (search snippet: https://www.aajjo.com/product/center-gondola-racks-in-surat-prayosha-steel-industries). An Australian range lists 300/375/450 mm deep x 900 mm and 300 x 1200 mm shelves (snippet). Suppliers say dimensions are customizable and bottom shelves are deeper.
   - **No Indian standard body specification was found.** Shelf-level spacing is adjustable, so it is not a reliable reference.
   - **[reasoning]** Bay width between uprights is the most reliable reference: it is large, so a 1-2 cm measurement error is about 1-2%. Strip height is small (39 mm), so a 3 mm mesh error is about 8%. Combine several references and check consistency. Best of all: one tape-measured reference per store, which gives ground truth.
2. **Per-image homography from shelf edges** **[reasoning]**. Two parallel strip lines plus known spacing give a planar rectification of the shelf front plane, which is accurate for the frontal width of a gap. It ignores depth (pushed-back products) and cannot fix lens distortion: arXiv 2205.13060 notes fish-eye and lens-position distortions "can't be corrected with planar projection" (https://arxiv.org/html/2205.13060). This makes a useful cross-check per frame.
3. **Monocular metric depth** (Depth Pro, MoGe-2, Metric3D, DA-V2-Small-metric). Gives absolute scale with no reference, but scale accuracy on close-up shelves is unverified (no source found). Metric3D states that focal length must be right. Use it only as a sanity check or to detect gross scale errors, not as the primary scale **[reasoning]**.

## 6. Failure modes

| Failure mode | Effect | Source / reasoning |
|---|---|---|
| Visible background through open shelf backs | Background products confuse the detector | Šikić 2024 |
| Products pushed back ("front" OOS) | Missed by product-detection methods | Šikić 2024; arXiv 2510.16508 |
| Gaps between adjacent products, hanging products at shelf ends | False positives | Šikić 2024 |
| Large top-shelf gaps | Under-localized | Šikić 2024 |
| Poor alignment, irregular shapes, shoppers in frame, lens distortion | Errors | arXiv 2205.13060 |
| Dark or black shelf backs, glossy/transparent bottles, motion blur, many identical packs | Weak or ambiguous texture breaks photogrammetry (holes or floating surfaces) and can produce false geometric gaps; repeated packs also cause wrong matches across frames | **[reasoning]**, not sourced. Mitigations: drop blurred frames (Laplacian variance), mask reflective regions, require 2D and 3D cues to agree |
| Glare and shadows | False positives | Roboflow tutorial (https://roboflow.com/blog/automate-on-shelf-monitoring) |
| Intentional display gaps vs real OOS | Pre-trained models struggle to tell them apart | Industry note via search snippet (Infosys blog, unverified) |

## 7. Recommendation for this project

**Hybrid: structure from shelf strips, gaps from 3D depth behind the front plane, confirmed by a fine-tuned 2D gap detector, with scale from fixtures.**

1. **Rows/slots (c):** detect yellow price strips on frames or renders, using Grounding DINO "price tag strip" plus SAM, or colour segmentation, then lift to 3D. This gives each row's front edge line and front plane.
2. **Scale:** scale the mesh from the bay width between uprights (900/1200 mm candidates) plus the strip height (about 39 mm candidate, to be confirmed in-store). Flag a scan if the two disagree by more than about 5% **[reasoning]**.
3. **Gap candidates (e):** run a per-row depth profile behind the front plane on the mesh, or on rendered depth from the existing renderer. Gaps get width and height in cm directly, and the depth class (fully empty vs "front"/pushed back) follows naturally.
4. **2D confirmation (a):** fine-tune **RF-DETR Small/Medium (Apache-2.0)** on gap classes {empty, front}, plus optionally product and price_tag. Start from CC BY / CC0 data (check each licence) and add our own frames. Avoid SKU-110K and Ultralytics unless licences are settled. Run it on frames, lift with the existing back-projection and DBSCAN, and keep gaps where the 2D and 3D evidence agree.
5. Run Grounding DINO "empty shelf" as a zero-cost baseline only.

**Rationale:**
- Direct detection has the best published 2D numbers (Šikić 2024), but there is no public data from Indian stores.
- 3D depth gives centimetres and the "front" OOS class for free, and fits the existing mesh pipeline.
- Licences stay commercial-safe: RF-DETR S/M, SAM/SAM2, Grounding DINO, MoGe-2 and DA-V2-Small are permissive. SKU-110K, DA-V2 B/L and Ultralytics are not, without extra licensing.

**Labelled data needed for evaluation:**
- (i) Per aisle, a gap list with tape-measured width and height in cm, plus row ID, ideally with a photo. This is the ground truth for cm error.
- (ii) In-store measurements of bay width, strip height, shelf depth and the spacing between shelf levels, to validate the scale assumptions.
- (iii) 2D box labels on about 300-1,000 sampled frames, using the Jha et al. rules (front face of the empty volume; one box per continuous gap; at least half a facing wide), with classes empty and front, plus price_tag. Jha's learning curve had not saturated at 800 images.
- (iv) Hard negatives: full shelves, intentional display gaps, dark backs, bottles.
- Metrics: gap-level precision/recall (IoU or centre-in-gap), mean absolute width error (cm), and the "front"-vs-empty confusion matrix.

---

## Sources (opened)

- Šikić et al. 2024, Sensors: https://pmc.ncbi.nlm.nih.gov/articles/PMC10819825/
- OOS-DSD: https://arxiv.org/abs/2510.16508 , https://arxiv.org/html/2510.16508
- Jha et al.: https://arxiv.org/abs/2205.13060 , https://arxiv.org/html/2205.13060
- SKU-110K: https://arxiv.org/abs/1904.00853 (PDF v3, stats from Table 1 and Sec. 4), https://github.com/eg4000/SKU110K_CVPR19 , https://docs.ultralytics.com/datasets/detect/sku-110k (search)
- OVDEval: https://arxiv.org/abs/2308.13177
- Planogram: https://arxiv.org/abs/2212.01004
- HiL dataset: https://humansintheloop.org/resources/datasets/supermarket-shelves-dataset/
- Roboflow blogs: https://roboflow.com/blog/retail-store-object-detection , https://roboflow.com/blog/automate-on-shelf-monitoring
- Roboflow Universe pages (403; snippets only): listed in section 3
- Grounding DINO: https://github.com/IDEA-Research/GroundingDINO ; HF API tags for IDEA-Research/grounding-dino-tiny and -base
- SAM: https://github.com/facebookresearch/segment-anything ; SAM2: https://github.com/facebookresearch/sam2
- RF-DETR: https://github.com/roboflow/rf-detr
- Ultralytics licence: https://www.ultralytics.com/license
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2 and metric_depth/README.md (raw)
- Metric3D: https://github.com/YvanYin/Metric3D
- MoGe: https://github.com/microsoft/MoGe ; HF API tags for Ruicheng/moge-2-vitl-normal and -vits-normal (MIT)
- Depth Pro: https://github.com/apple/ml-depth-pro and its LICENSE (raw)
- HL Display datastrips: https://www.hl-display.com/catalogue/price-communication/datastrips-for-paper-labels/antl
- IndiaMART rack listings: https://dir.indiamart.com/bengaluru/supermarket-display-rack.html ; https://www.aajjo.com/product/big-bazaar-supermarket-gondola-rack
- Milella et al. (search summary only): https://doaj.org/article/302301b555c1481b8a0ef957710e06f5 , https://iris.cnr.it/handle/20.500.14243/446128
- Shelf-edge patent: https://patents.google.com/patent/US11200677 (search only)
