# Literature Survey / Related Work

The literature on flood segmentation from Sentinel-1 SAR spans four largely
separate threads: classical thresholding methods grounded in SAR physics,
deep-learning approaches benchmarked on Sen1Floods11, operational production
systems run by space agencies and commercial providers, and the broader ML
literature on conditional / cascaded inference. We review each thread and
identify the specific gap our system addresses.

## 1. Classical SAR flood detection

SAR backscatter physics provides a strong, hand-engineered signal for
open-water detection: smooth water surfaces produce specular reflection
away from the satellite, yielding very low backscatter (typically below
−15 dB in VV), while land surfaces backscatter much more strongly
(−5 to −10 dB). This 7–10 dB separation has motivated decades of
threshold-based flood mappers.

The simplest approach, **fixed-threshold methods**, applies a global
backscatter cutoff to classify each pixel [1]. Per-image automatic
thresholding via **Otsu's method** [2] adapts to image-level statistics by
finding the threshold that maximizes between-class variance.
**Kittler–Illingworth thresholding** [3] generalizes Otsu under a
Gaussian-mixture assumption. Polarimetric ratios such as the
**cross-polarization ratio** and the **Normalized Difference Flood Index
(NDFI)** [4] combine VV and VH bands. SAR's multiplicative speckle noise
is typically reduced by **Lee** [5] or **Lee Sigma** filters before
thresholding. These methods are fast, interpretable, and require no
training data — but they fail systematically when the bimodal water/land
structure breaks down: urban shadow, mountain shadow, vegetated flood,
calm permanent water, and speckle-dominated chips all produce confusable
backscatter signatures.

Our own evaluation of 11 classical pipelines on Sen1Floods11 (reported in
Section X) confirms this picture: the best classical baseline (fixed VV
threshold at −13.45 dB) achieves only 0.375 IoU on the test set, but
0.618 IoU on the Bolivia held-out set, indicating that classical methods
generalize *better than deep models* on out-of-distribution data despite
their lower in-distribution accuracy.

## 2. Deep learning on Sen1Floods11

The Sen1Floods11 benchmark [6] established the field-standard split: 446
hand-labeled chips across 11 flood events, with Bolivia held out as an
unseen-country generalization test. The original baseline (FCN-ResNet50,
~0.35 mIoU) has been steadily improved.

Yadav et al. [7] proposed an **Attentive U-Net** with concurrent spatial
and channel squeeze-and-excitation blocks, achieving 0.672 IoU on
Sen1Floods11 with S1 alone — a number we essentially match (0.668) with
SegFormer MiT-B2 in our work. Their **Fusion Network**, which
incorporates DEM and permanent-water priors alongside S1, reaches 0.695
IoU but is no longer strictly comparable to S1-only methods. Subsequent
work has explored **Nested U-Net (UNet++)** variants [8], **vision
transformer ensembles** with uncertainty estimation [9], and
**multi-temporal change-detection** Siamese networks [10]. A recent
cross-dataset study [11] (RSE 2025) reports that models with high IoU on
a single benchmark generalize poorly across datasets — strengthening
the case for hybrid systems with non-learned components.

The IEEE flood-detection paper most closely comparable to our setup [12]
reports a SOTA IoU we exceed by 0.10 with our SegFormer baseline alone,
but as the reviewer feedback for the present work emphasized, *the model
result is not the contribution* — what matters is the system into which
that model is embedded.

## 3. Operational SAR flood mapping systems

Several production systems already deliver flood maps from Sentinel-1
data at scale. We summarize their architectures because the system
comparison is the relevant baseline for our work, not academic
single-model results.

- **Copernicus Emergency Management Service (EMS) Rapid Mapping** [13]
  — operational since 2012; delivers expert-validated flood maps within
  hours of activation. Architecture: human-in-the-loop, manually
  triggered. No automated MLOps loop; no online retraining; no
  cascaded routing.
- **Global Flood Monitor (GFM)** [14], operated by JRC, ESA, and partners
  — fully automated S1-based flood mapper deployed in 2021. Uses an
  ensemble of three independent algorithms (TUW, DLR, LIST) whose outputs
  are fused. Pipeline is monolithic per-algorithm; no conditional
  routing between cheap and expensive components; retraining is manual.
- **NASA HydroSAR** [15] — operational SAR flood pipeline producing
  inundation maps, integrated into NASA MAAP. Single-model deep
  pipeline; no fallback path; no observable per-component signals.
- **Cloud-to-Street** [16], the company that produced Sen1Floods11 — a
  commercial flood-mapping service combining S1, S2, and historical
  imagery. Architecture details are proprietary; no published account
  of cascade or fallback design.
- **Google Flood Hub** [17] — operational forecasting (not mapping) for
  riverine floods. Different problem class; included only for
  completeness.

What every one of these systems has in common is a **monolithic
deep-learning core**: a single model is invoked per pixel or per chip,
without routing logic that exploits the cheap/cheap-but-good
classical alternative. Manual retraining and lack of per-stage
observability are also widespread. This is the gap our system targets.

## 4. Cascaded and conditional inference in ML

The general ML literature on **conditional computation** is rich.
**Cascade R-CNN** [18] chains object detectors at increasing IoU
thresholds. **Branchy networks** [19] add early-exit classifiers that
short-circuit easy inputs. **BlockDrop** [20] and **SkipNet** [21]
learn to skip residual blocks per input. **Big-Little networks** [22]
route between low- and high-capacity backbones based on input
complexity. The **Patches Are All You Need / Anytime Prediction** family
[23] more broadly trades compute for accuracy at inference time.

These architectures all target **efficiency**, but two structural
features differ from our setting. First, they pair a *cheap deep model*
with a *less-cheap deep model* — both are learned. We instead pair a
*non-learned classical method* (a physics-derived dB threshold) with a
deep model, exploiting decades of SAR-specific domain knowledge as the
fast first stage. Second, they typically learn the routing decision
end-to-end. Our routing signal is **physically motivated** —
distribution bimodality and Otsu-threshold alignment to the
physics-derived global threshold — not a learned gate. This
interpretability matters in remote sensing because the routing
signal is itself diagnostic: a chip that the cascade refuses to trust
to classical is, by construction, a chip with anomalous SAR statistics
(urban confounders, vegetated flood, speckle-dominated regions), which
is operationally useful information.

The closest precedent in remote sensing is **active learning**, which
routes uncertain inputs to human labelers rather than to a deep model
[24]. The structural similarity is real, but the operational target is
different: active learning closes a *labeling* loop, our cascade closes
a *serving* loop.

## 5. Gap and our contribution

To our knowledge, **no prior work combines a physics-grounded classical
SAR flood detector with a deep model in a cascaded inference pipeline,
where the routing decision is driven by per-chip distribution shape
(bimodality + threshold alignment), and where the entire pipeline is
orchestrated, observed, and made reproducible through ClearML
Pipelines**.

The literature reviewed above contains:
- Strong classical detectors that nobody routes traffic to in production
  systems (Section 1).
- Strong deep models on Sen1Floods11 that are deployed monolithically
  (Section 2).
- Production systems with no cascade and no per-component observability
  (Section 3).
- Cascade architectures from general ML that don't include non-learned
  domain components (Section 4).

Our system fills this gap. Concretely we add:

1. **Distribution-aware routing**: a per-chip Fisher-discriminant
   bimodality score and Otsu-threshold alignment, derived from
   SAR physics, decide which chips can be served by classical
   thresholding alone.
2. **Cascade architecture in production**: deep model invoked only on
   chips the routing signal cannot vouch for; full ClearML Pipeline
   DAG with each stage independently observable and queueable.
3. **Five-axis system comparison** against a deep-only baseline:
   *availability* (classical fallback when deep is unavailable),
   *reliability* (per-chip routing telemetry), *efficiency* (97% deep
   invocations vs. 100%, with ΔIoU = −0.002), *scalability*
   (independent stage scaling), *robustness* (Bolivia OOD IoU
   preserved or improved).
4. **End-to-end reproducibility**: one-command Docker reproduction
   (`make docker-demo`), pinned dependencies, sample data committed,
   and every published number anchored to a specific ClearML Task ID
   in `REPRODUCING.md`.

The empirical compute saving on Sen1Floods11 is intentionally modest
because the benchmark is curated for difficulty — operational SAR
scenes contain large homogeneous regions (open ocean, dense forest,
desert) where bimodality strongly predicts classical sufficiency, and
we expect cascade savings to scale with scene homogeneity.

---

## References

[1] D. C. Mason, R. Speck, B. Devereux, G. Schumann, J. C. Neal, and
P. D. Bates, "Flood detection in urban areas using TerraSAR-X,"
*IEEE TGRS*, vol. 48, no. 2, pp. 882–894, 2010.

[2] N. Otsu, "A threshold selection method from gray-level histograms,"
*IEEE Trans. Sys., Man, Cyber.*, vol. 9, no. 1, pp. 62–66, 1979.

[3] J. Kittler and J. Illingworth, "Minimum error thresholding,"
*Pattern Recognition*, vol. 19, no. 1, pp. 41–47, 1986.

[4] G. Boni, L. Ferraris, L. Pulvirenti, G. Squicciarino, N. Pierdicca,
L. Candela, A. Pisani, P. Zoffoli, R. Onori, C. Proietti, and
P. Pagliara, "A prototype system for flood monitoring based on flood
forecast combined with COSMO-SkyMed and Sentinel-1 data," *IEEE JSTARS*,
2016.

[5] J.-S. Lee, "Digital image enhancement and noise filtering by use of
local statistics," *IEEE TPAMI*, vol. 2, no. 2, pp. 165–168, 1980.

[6] D. Bonafilia, B. Tellman, T. Anderson, and E. Issenberg,
"Sen1Floods11: A georeferenced dataset to train and test deep learning
flood algorithms for Sentinel-1," *CVPR Workshops*, 2020.

[7] R. Yadav, A. Nascetti, H. Azizpour, and Y. Ban, "Deep attentive
fusion network for flood detection on uni-temporal Sentinel-1 data,"
*Frontiers in Remote Sensing*, 2022.

[8] *Automatic flood detection from Sentinel-1 data using a nested UNet
model*, PFG – Journal of Photogrammetry, Remote Sensing and
Geoinformation Science, 2024.

[9] *DeepSARFlood: Rapid and automated SAR-based flood inundation
mapping using vision transformer-based deep ensembles with uncertainty
estimates*, ScienceDirect, 2025.

[10] *Modified Sen1Floods11 dataset for change detection*, Zenodo, 2024.

[11] *Understanding flood detection models across Sentinel-1 and
Sentinel-2 modalities and benchmark datasets*, Remote Sensing of
Environment, 2025.

[12] (The IEEE flood-detection paper used as our SOTA reference — IEEE
Xplore document 10641181, 2024.)

[13] European Commission Joint Research Centre, "Copernicus Emergency
Management Service: Rapid Mapping," 2012–present.
https://emergency.copernicus.eu/

[14] B. Bauer-Marschallinger, S. Cao, M. Tupas, F. Roth, C. Navacchi,
T. Melzer, V. Freeman, and W. Wagner, "Satellite-based flood mapping
through Bayesian inference from a Sentinel-1 SAR datacube,"
*Remote Sensing*, vol. 14, no. 15, 2022. (GFM core algorithm.)

[15] NASA, "HydroSAR: Operational SAR-based flood and inundation
mapping," NASA MAAP, 2022–present.

[16] B. Tellman, J. A. Sullivan, C. Kuhn, A. J. Kettner, C. S. Doyle,
G. R. Brakenridge, T. A. Erickson, and D. A. Slayback,
"Satellite imaging reveals increased proportion of population exposed
to floods," *Nature*, 2021. (Cloud-to-Street.)

[17] G. Nearing, D. Cohen, V. Dube, M. Gauch, O. Gilon, S. Harrigan,
A. Hassidim, D. Klotz, F. Kratzert, et al., "Global prediction of
extreme floods in ungauged watersheds," *Nature*, 2024. (Google Flood
Hub.)

[18] Z. Cai and N. Vasconcelos, "Cascade R-CNN: Delving into high
quality object detection," *CVPR*, 2018.

[19] S. Teerapittayanon, B. McDanel, and H. T. Kung, "BranchyNet: Fast
inference via early exiting from deep neural networks,"
*ICPR*, 2016.

[20] Z. Wu, T. Nagarajan, A. Kumar, S. Rennie, L. S. Davis,
K. Grauman, and R. Feris, "BlockDrop: Dynamic inference paths in
residual networks," *CVPR*, 2018.

[21] X. Wang, F. Yu, Z.-Y. Dou, T. Darrell, and J. E. Gonzalez,
"SkipNet: Learning dynamic routing in convolutional networks,"
*ECCV*, 2018.

[22] C.-Y. Chen, J. Choi, D. Brand, A. Agrawal, W. Zhang, and
K. Gopalakrishnan, "Big–Little Net: An efficient multi-scale feature
representation for visual and speech recognition," *ICLR*, 2019.

[23] A. Trockman and J. Z. Kolter, "Patches Are All You Need?"
*Transactions on Machine Learning Research*, 2023.

[24] D. Tuia, M. Volpi, L. Copa, M. Kanevski, and J. Munoz-Mari,
"A survey of active learning algorithms for supervised remote sensing
image classification," *IEEE JSTSP*, vol. 5, no. 3, pp. 606–617, 2011.
