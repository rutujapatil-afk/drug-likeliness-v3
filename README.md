# Drug-Likeliness V3

Reproducible computational study of structure-based molecular classification,
with emphasis on endpoint provenance, validation robustness, chemical-space
dependence, and external transportability.

## Project status

The computational analyses are complete and the principal results are
frozen. The project is currently in the manuscript and publication-analysis
phase.

The final workflow includes:

- dataset construction and quality control
- molecular standardization and deduplication
- source and overlap auditing
- molecular fingerprint generation
- random-split baseline evaluation
- scaffold-separated evaluation
- repeated scaffold evaluation across five seeds
- logistic-regression comparison
- descriptor-calibrated sensitivity analysis
- bootstrap uncertainty estimation
- chemical-space applicability-domain analysis
- systematic error analysis
- external validation on ClinTox
- publication evidence generation
- publication figure generation

The computational results should be treated as frozen unless a final
reproducibility audit identifies a methodological or implementation error.

---

## Scientific objective

The study asks whether a molecular machine-learning model can discriminate
between two operationally defined molecular populations and, more
importantly, whether that discrimination remains stable when validation
becomes more stringent.

The central scientific question is not whether the model has discovered a
universal molecular property called "drug-likeness." Instead, the study
examines how much of the observed discrimination persists under scaffold
separation, descriptor balancing, chemical-space analysis, and an external
endpoint shift.

---

## Research questions

### RQ1
Does discrimination persist under repeated scaffold-separated evaluation?

### RQ2
Does discrimination remain after reducing measured physicochemical
descriptor imbalance, and how does performance vary across chemical-space
support?

### RQ3
Does the learned signal transfer to an independently defined external
endpoint?

### Hypothesis

We hypothesized that internal discrimination would remain above chance under
robustness controls but would weaken as chemical-space and endpoint
distributions diverged.

---

## Development dataset

The final modeling dataset contains 8,196 unique standardized molecules:

| Source | Label 0 | Label 1 | Total |
|---|---:|---:|---:|
| ChEBI-derived negative-reference population | 4,098 | 0 | 4,098 |
| DrugCentral-derived positive population | 0 | 4,098 | 4,098 |
| **Total** | **4,098** | **4,098** | **8,196** |

Structures were standardized and deduplicated before modeling.

### Critical endpoint limitation

The development label is perfectly confounded with source membership:

- ChEBI-derived negative-reference population → label 0
- DrugCentral-derived positive population → label 1

Consequently, the development task should be interpreted as discrimination
between these constructed reference populations.

It should not be interpreted as direct validation of a universal molecular
property of "drug-likeness."

This source-derived endpoint terminology is retained throughout the project
to make the provenance of the labels explicit.

---

## Molecular representation

The primary molecular representation is a:

- 2,048-bit Morgan fingerprint
- radius = 2

Fingerprints are generated from standardized canonical SMILES using RDKit.

The primary classifier is a Random Forest with:

- 500 trees
- class-balanced training
- deterministic random seeds
- scikit-learn implementation

Logistic regression is additionally used as a simpler model comparison in
the repeated scaffold analysis.

---

## Evaluation strategy

The evaluation hierarchy becomes progressively more conservative:

1. conventional random train/validation/test split
2. scaffold-separated evaluation
3. repeated scaffold evaluation across five seeds
4. descriptor-calibrated sensitivity analysis
5. chemical-space applicability-domain analysis
6. external endpoint-shift evaluation

The scaffold-separated and sensitivity analyses are emphasized over the
conventional random split when interpreting generalization.

---

## Main results

### Conventional random split

The conventional held-out test set produced:

- ROC-AUC: 0.9164
- PR-AUC: 0.9001
- Accuracy: 0.8488
- F1: 0.8588

These results demonstrate strong discrimination within the constructed
development task but are not by themselves evidence of a universally
transferable drug-likeness property.

---

### Repeated scaffold evaluation

Five scaffold-separated seeds produced:

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Random Forest | 0.8924 ± 0.0219 | 0.8775 ± 0.0283 |
| Logistic regression | 0.8518 ± 0.0187 | 0.8454 ± 0.0158 |

The Random Forest therefore retained substantial discrimination after
scaffold separation, although performance was lower than in the
conventional random split.

---

### Descriptor-calibrated sensitivity analysis

A stringent descriptor-based caliper-matching procedure was used to reduce
measured physicochemical imbalance between the two source-derived
populations.

Across five seeds:

- ROC-AUC: 0.8745 ± 0.0227
- PR-AUC: 0.8674 ± 0.0254

The matching procedure retained approximately 40–43% of the positive
population per seed, with median matching distances of approximately 0.40
in the standardized descriptor space.

Relative to the corresponding repeated scaffold results:

- paired ROC-AUC difference: −0.0178
- paired 95% interval: −0.0221 to −0.0136
- paired PR-AUC difference: −0.0100
- paired 95% interval: −0.0210 to 0.0010

The results therefore indicate a modest reduction in discrimination after
stronger descriptor balancing rather than complete disappearance of the
signal.

---

## Uncertainty estimation

Bootstrap uncertainty was estimated using 5,000 bootstrap replicates.

For the locked scaffold-separated test evaluation:

- ROC-AUC: 0.8771
- bootstrap 95% CI: 0.8515–0.9013
- PR-AUC: 0.8503
- bootstrap 95% CI: 0.8113–0.8867

Bootstrap confidence intervals are reported separately from the
across-seed variability of the repeated scaffold experiments.

The repeated-scaffold values describe variation across independently
constructed scaffold partitions, whereas bootstrap intervals quantify
sampling uncertainty conditional on a particular evaluated test set.

---

## Chemical-space applicability domain

Chemical-space support was assessed using the maximum Tanimoto similarity
between each test molecule and the training population.

The molecular fingerprints used for this analysis were the same
2,048-bit radius-2 Morgan fingerprints used for the primary model.

Performance varied strongly with chemical-space proximity:

| Maximum training-set Tanimoto similarity | ROC-AUC | PR-AUC |
|---|---:|---:|
| <0.4 | 0.7602 | 0.6632 |
| 0.4–0.5 | 0.8551 | 0.8608 |
| 0.5–0.6 | 0.8696 | 0.8522 |
| 0.6–0.7 | 0.9480 | 0.9621 |
| 0.7–0.8 | 0.9436 | 0.9456 |
| 0.8–0.9 | 0.9931 | 0.9762 |
| ≥0.9 | 1.0000 | 1.0000 |

These results indicate that predictive discrimination is not chemically
uniform.

Performance was substantially weaker for molecules farther from the
training chemical space and stronger for molecules with greater structural
similarity to the training population.

The applicability-domain analysis therefore provides an important
qualification on the reported internal performance: predictions should not
be assumed to have equivalent reliability across chemical space.

---

## External validation

External validation was performed using ClinTox after:

1. structure standardization
2. exclusion of invalid/query structures
3. within-external deduplication
4. removal of molecules overlapping the development dataset

The external dataset contained:

- 934 final molecules
- 893 FDA-approved molecules
- 41 non-approved molecules

The model produced:

- ROC-AUC: 0.4701
- bootstrap 95% CI: 0.3871–0.5549
- PR-AUC: 0.9518

The ClinTox positive-class prevalence was 0.9561.

Therefore, although PR-AUC remained numerically high, it was slightly below
the positive-class prevalence. In this highly imbalanced external dataset,
PR-AUC should therefore not be interpreted as evidence of strong
transportability.

The ROC-AUC near 0.5 indicates that the model did not retain useful ranking
ability for the external endpoint.

### Endpoint shift

The ClinTox evaluation does not reproduce the development endpoint.

The development task distinguishes a DrugCentral-derived positive population
from a ChEBI-derived negative-reference population, whereas the external
evaluation uses an FDA-approval endpoint.

The ClinTox experiment should therefore be interpreted as an external
endpoint-shift and transportability stress test rather than as a direct
replication of the development classification task.

The external result supports the conclusion that strong internal
discrimination does not necessarily transfer to an independently defined
molecular endpoint.

---

## Interpretation

The combined analyses support three conclusions.

First, substantial discrimination exists between the constructed
source-derived reference populations.

Second, much of that discrimination persists under stricter validation,
including scaffold separation and descriptor-calibrated sensitivity analysis.

Third, the signal is strongly dependent on chemical-space proximity and
does not transfer to the independently defined ClinTox endpoint.

Taken together, the findings support reproducible discrimination of the
constructed reference populations but do not support interpreting that
discrimination as a universally transferable molecular property of
drug-likeness.

---

## Reproducibility

The computational workflow is organized into reusable Python package code
under `src/` and executable analysis scripts under `scripts/`.

Important project documentation includes:

- `DATASET_DEFINITION.md`
- `DATASET_PROTOCOL.md`
- `RESEARCH_MANIFEST.md`
- `README.md`

The research manifest maps experiments to their corresponding scripts and
outputs.

Publication figures are generated using:

```text
scripts/plot_publication_figures.py