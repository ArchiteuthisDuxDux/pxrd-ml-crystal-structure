# Machine Learning for Powder X-Ray Diffraction Analysis

## Overview

This project develops a practical machine-learning pipeline that takes an experimental powder X-ray diffraction (PXRD) pattern and produces preliminary information about the crystal structure.

The goal is not to replace crystallographic refinement or introduce a new neural-network architecture. The project focuses on the complete workflow:

- collecting and inspecting data from multiple sources;
- handling missing, inconsistent, and incomplete labels;
- cleaning data and converting it to a common format;
- training a model when experimental data are scarce;
- evaluating the model without train/validation leakage;
- testing plausible approaches and preserving negative results.

The final model predicts unit-cell parameters, crystal system, space group, and the chemical elements present in a sample. It is a fast screening tool, not a substitute for full crystallographic analysis.

## What is PXRD?

In PXRD, a powdered sample is exposed to X-rays and measured at different angles. The result is a one-dimensional profile: the horizontal axis is the scattering angle `2θ`, the vertical axis is intensity, and diffraction peaks arise from constructive interference of waves scattered by atoms.

Peak positions are related to unit-cell dimensions. The peak pattern and relative intensities also depend on the atoms and symmetry of the structure. The inverse problem is difficult: peaks overlap, background and noise depend on the instrument and sample preparation, and different structures can produce similar profiles. The model should therefore be treated as a source of candidates for further verification.

## Task definition

The model receives:

- intensity sampled on a common grid of `4096` points from `0°` to `90°`;
- a validity mask indicating which points were actually measured;
- the primary and secondary X-ray wavelengths.

It predicts:

1. Unit-cell edges `a`, `b`, and `c`, plus angles `α`, `β`, and `γ`.
2. Crystal system, one of the seven conventional crystal-system classes.
3. Space group, one of the 230 crystallographic space groups.
4. The set of chemical elements present in the sample.
5. `log(V)`, the logarithm of unit-cell volume, as an auxiliary target.

Volume is calculated from the lattice parameters and is not an independent physical parameter.

## Related work

Classical PXRD analysis combines peak finding, indexing, database search, and profile refinement, for example with [Rietveld refinement](https://en.wikipedia.org/wiki/Rietveld_refinement). This remains the reliable route, while the neural network accelerates the preliminary stage.

| Evaluation setting | Space-group top-1 |
|---|---:|
| ICSD, random split | 83.2% |
| ICSD, structure-type split | 55.9% |
| Synthetic → calculated ICSD, ResNet-101 | 79.9% |
| Distorted synthetic → experimental **RRUFF** | 25.2% |

The drop from 83.2% to 55.9% shows how random splits can overestimate performance when related structures appear in both partitions. The same issue was found independently in this project.

## Data

### **RRUFF**

[**RRUFF**](https://rruff.info/) is an open mineralogical project containing powder XRD, Raman, and infrared spectra, chemical compositions, and sample metadata. After filtering by ID, the local subset contained 1,359 spectra: 1,298 with lattice parameters, 1,357 with chemical composition, and 1,298 with crystal-system labels. The initial archive contained no space-group labels; these were later reconstructed from the official IMA export.

### **opXRD**

[**opXRD**](https://advanced.onlinelibrary.wiley.com/doi/10.1002/aidi.202500044) is an open compilation of experimental powder XRD patterns. The original publication reports 92,552 spectra, including 2,179 with at least some structural annotations. The archive is available on [Zenodo](https://zenodo.org/records/14279434).

opXRD combines laboratories in France, China, Switzerland, the United States, and Germany. Its spectra therefore differ in range, step size, wavelength, instrument response, background, and noise.

After duplicate removal, 88,484 spectra remained: 2,116 entered the fine-tuning pool, 86,368 lacked sufficient structural information, 570 had a space-group label, 869 had element labels, and approximately 1,355 had lattice and crystal-system labels. For approximately 180 labeled rows with a missing primary wavelength, `Cu Kα = 1.5406 Å` was imputed.

| Property | **RRUFF** | **opXRD** |
|---|---|---|
| Main materials | Minerals | Multiple material classes |
| Acquisition | Relatively standardized | Many laboratories and instruments |
| Mean number of elements | 4.69 | 2.45 |
| Median `a` | 8.81 Å | 6.16 Å |
| Peak width | Generally narrower | Generally broader |
| Background | Usually weaker | Usually more pronounced |

A diagnostic classifier distinguished not only synthetic from real data, but also RRUFF from opXRD. This indicates both a synthetic-to-real gap and a real-to-real domain gap.

### Crystal-structure sources for synthetic data

The pretraining stage requires hundreds of thousands of examples with precise labels. [Crystallography Open Database](https://www.crystallography.net/cod/) supplied 324,875 filtered structures. In this project, `crystalDB` refers to a local collection of 135,258 structures from the [Materials Project](https://next-gen.materialsproject.org/); its space groups were calculated separately.

## From CIF to XRD

CIF, or Crystallographic Information File, is a text format describing a crystal structure. It typically contains unit-cell parameters, atomic coordinates, occupancies, space-group information, and publication metadata.

A CIF does not contain a ready-made XRD profile. The generator reads the structure, enumerates `hkl` planes, calculates reflection angles, applies IT92 atomic form factors, sums phase-dependent atomic contributions, and accounts for equivalent reflections and geometric corrections.

This is standard diffraction physics, not a full DFT calculation. The in-house generator was checked against `pymatgen`: peak positions agreed, full-profile correlations were typically `0.97–0.997`, and the vectorized implementation was approximately ten times faster.

## Making ideal patterns look experimental

| Component | Simulated effect |
|---|---|
| Kα doublet | Two nearby laboratory X-ray lines |
| March–Dollase model | Preferred crystallite orientation |
| Random peak scaling | Sample-dependent relative intensities |
| Caglioti relation | Angle-dependent instrumental broadening |
| Scherrer relation | Broadening from small crystallite size |
| Microstrain | Additional lattice broadening |
| Pseudo-Voigt profile | Gaussian/Lorentzian peak shape |
| Zero shift and sample displacement | Peak shifts |
| Amorphous halo | Broad amorphous signal |
| Poisson noise | Intensity-dependent counting noise |
| Saturation and zingers | Saturation and isolated outliers |

RRUFF-like synthetic data use narrower peaks, weaker backgrounds, and an approximately 10% halo probability. opXRD-like synthetic data use broader peaks, stronger backgrounds, and an approximately 18% halo probability.

$$
M = (2\theta_{\min}, 2\theta_{\max}, \Delta 2\theta, \lambda_{K\alpha_1}, \lambda_{K\alpha_2})
$$

$$
X_{\mathrm{syn}} = G(\mathrm{CIF}, M, \boldsymbol{\xi}),
$$

where `ξ` contains random experimental-distortion parameters.

## Data preparation

Every profile is converted to a common Parquet schema, checked for acquisition mode and wavelength ordering, and resampled to `(2, 4096)`. The first channel contains the square root of normalized intensity; the second contains the valid-point mask.

| Partition | Rows |
|---|---:|
| Synthetic pretraining | 460,133 |
| Real with at least one label | 3,475 |
| Real without required labels | 86,368 |

| Head | Labeled rows |
|---|---:|
| Lattice and volume | 2,653 |
| Crystal system | 2,684 |
| Space group | 1,675 |
| Elements | 2,253 |

## Model and training

The model is a one-dimensional residual CNN with a shared trunk and five task-specific heads: lattice, volume, space group, crystal system, and elements. It has approximately 11.3 million parameters, six residual blocks with stride 2, GroupNorm, AdamW, cosine scheduling, warmup, gradient clipping, and mixed-precision training.

Each loss is computed only when the corresponding label exists. Smooth L1 is used for lattice and volume, cross-entropy for space group and crystal system, and binary cross-entropy for elements.

| Version | Experiment | Outcome |
|---|---|---|
| V1 | Small CNN, 1.37M parameters | Working baseline |
| Pretrain V2 | 11.3M CNN, 460,133 synthetic profiles | Main synthetic backbone |
| FT V2 | Real data + 50/50 synthetic replay | First complete result |
| Real-only V2 | Trained from scratch on real data | Synthetic ablation |
| Staged V3 | Replay followed by real-only training | No overall improvement |
| Source-only | Separate RRUFF and opXRD models | Domain-gap analysis |
| Combined, no replay | Fine-tuning on both real sources | Main scheme |
| `with_rruff_sg` | New RRUFF labels and mineral-aware split | Final cycle |

Final fine-tuning starts from `pretrain_v2_full_best.pt`, uses no synthetic replay, and runs for 20 real-data epochs.

## Recovering RRUFF space-group labels

The original RRUFF archive had spectra, IDs, mineral names, formulas, and lattice parameters, but no dedicated space-group field. The official IMA export contained names and space groups, while its RRUFF-ID column was empty for all 6,228 minerals.

The conservative linkage procedure was:

1. Match an RRUFF ID if available.
2. Otherwise match the exact mineral name and plain name.
3. Normalize case, hyphens, and diacritics.
4. Convert [Hermann–Mauguin notation](https://en.wikipedia.org/wiki/Hermann%E2%80%93Mauguin_notation) to numbers 1–230 with Gemmi.
5. Map equivalent settings, such as `Pbnm` and `Pnma`, to the same group.
6. Use the known crystal system only to resolve a remaining choice.
7. Exclude unresolved, conflicting, and genuinely ambiguous rows.

For 1,359 unique RRUFF IDs, this produced **1,105 space-group labels**, covering **81.3% of RRUFF**, 661 minerals, and 93 space groups. A terminology issue was also identified: RRUFF sometimes calls a trigonal structure `hexagonal` when it is written in the hexagonal axis setting. The original name is preserved, while the working crystal-system label is made consistent with the space-group number.

## Leakage-free evaluation

An initial random split produced space-group accuracy above 90%, but 90 of 174 validation rows belonged to compounds already present in training. The final protocol uses an outer five-fold `GroupKFold` for evaluation and an inner group-aware split for hyperparameter selection. RRUFF spectra from the same mineral stay in one fold; opXRD continues to use `conn_key`.

## Results

### Final RRUFF-only model

| Metric | Zero-shot | RRUFF fine-tuning |
|---|---:|---:|
| Crystal system | 42.8% | **57.6% ± 3.4%** |
| SG top-1 | 23.9% | **42.9% ± 2.4%** |
| SG top-5 | 55.1% | **68.2% ± 4.0%** |
| Elements F1 | 23.2% | **53.9% ± 1.8%** |
| MAE `a` | 2.91 Å | **2.61 Å** |
| Angle MAE | 5.91° | **5.52°** |

### Final combined model

| Metric | Zero-shot | Combined fine-tuning |
|---|---:|---:|
| Crystal system | 40.4% | **62.2% ± 2.7%** |
| SG top-1 | 17.9% | **42.2% ± 7.2%** |
| SG top-5 | 40.3% | **65.1% ± 5.1%** |
| Elements F1 | 17.8% | **54.8% ± 4.5%** |
| Elements exact | 0.6% | **20.4% ± 5.4%** |
| MAE `a` | 3.40 Å | **2.64 Å** |
| Median error `a` | 2.09 Å | **1.55 Å** |
| Angle MAE | 5.85° | **4.75°** |

Synthetic pretraining improved crystal-system accuracy by about 10 percentage points, space-group top-1 by 17.8 points, and top-5 by 14.2 points. Its main benefit is concentrated in the structural heads.

## Failed hypotheses

- **Pseudo-labeling:** strict validation left only 20 usable rows out of 86,368 unlabeled opXRD profiles, so pseudo-labels were not used.
- **Permanent replay:** real fine-tuning without synthetic replay performed better overall.
- **Larger models and longer training:** these produced only small gains.
- **q-space input:** the project keeps `2θ` as input and provides wavelength separately.
- **Removing rare space groups:** this would artificially improve accuracy.

## Limitations

- RRUFF space-group labels are assigned at the mineral-species level, not from an independent refinement of every spectrum.
- 189 RRUFF rows remain ambiguous and are excluded from the SG loss.
- Space-group classes remain imbalanced.
- opXRD combines many laboratories and has substantial fold-to-fold variation.
- Wavelength is imputed for part of opXRD.
- Carbon and hydrogen are weakly represented in conventional XRD.
- Impurities and multiphase samples complicate element and SG prediction.
- No third independent laboratory dataset is available for external validation.

## Reproducibility and checkpoints

The project provides cleaned datasets, a common representation, a physics-based generator, validation against `pymatgen`, a real/synthetic diagnostic classifier, 460,133 synthetic pretraining profiles, leakage-aware splits, RRUFF SG labels, experiment history, and final checkpoints.

Main checkpoint:

```text
ft_combined_no_replay_control_with_rruff_sg_final.pt
```

For data known to be opXRD-like, `ft_opXRD_only_final.pt` remains stronger on SG and element metrics. The combined checkpoint is the preferred general-purpose model.

## Future work

1. Add single-pattern inference with top-5 predictions and a confidence warning.
2. Repeat mineral-aware evaluation with the RRUFF SG loss disabled.
3. Add class weights or a balanced sampler for rare space groups.
4. Couple space-group and crystal-system predictions hierarchically.
5. Add a weak second crystalline phase to the synthetic generator.
6. Evaluate on a third independent laboratory source.

## Conclusion

The project establishes a reproducible pipeline from CIF files and heterogeneous experimental archives to a multi-task model that predicts unit-cell parameters, crystal system, space group, and chemical elements.

The key conclusions are:

- synthetic data are useful for pretraining, especially for structural tasks;
- permanent synthetic replay during fine-tuning is unnecessary;
- RRUFF and opXRD differ substantially even though both are real datasets;
- official archives should not be assumed to be complete or training-ready;
- linking IMA and RRUFF increased real SG labels from 570 to 1,675;
- stricter splits are more valuable than attractive but inflated accuracy numbers.

## References

1. [**RRUFF** Project](https://rruff.info/) and [RRUFF instrumentation](https://rruff.info/about/downloads/About_RRUFF_brochure.pdf).
2. Hollarek et al., [**opXRD**: Open Experimental Powder X-Ray Diffraction Database](https://doi.org/10.1002/aidi.202500044).
3. [**opXRD** archive on Zenodo](https://zenodo.org/records/14279434).
4. [Crystallography Open Database](https://www.crystallography.net/cod/).
5. [Materials Project](https://docs.materialsproject.org/).
6. [Materials Project: diffraction patterns](https://docs.materialsproject.org/methodology/materials-methodology/diffraction-patterns).
7. IUCr, [Crystallographic Information Framework](https://www.iucr.org/resources/cif).
8. IUCr, [Bragg peak profiles and pseudo-Voigt functions](https://journals.iucr.org/j/issues/2021/06/00/gj5272/index.html).
9. International Tables for Crystallography: [Scherrer equation](https://onlinelibrary.wiley.com/iucr/itc/Ha/ch5o1v0001/sec5o1o2/) and [preferred orientation](https://onlinelibrary.wiley.com/iucr/itc/Ha/ch2o10v0001/sec2o10o1/).
10. [IUCr powder CIF dictionary: Chebyshev background](https://www.iucr.org/resources/cif/dictionaries/browse/cif_pd).
11. Schopmans, Reiser, and Friederich, [Neural networks trained on synthetically generated crystals can extract structural information from ICSD powder X-ray diffractograms](https://doi.org/10.1039/D3DD00071K), [supplementary material](https://www.rsc.org/suppdata/d3/dd/d3dd00071k/d3dd00071k1.pdf), and [ML4pXRDs](https://github.com/aimat-lab/ML4pXRDs).
