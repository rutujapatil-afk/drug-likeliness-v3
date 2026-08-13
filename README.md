# Drug-Likeliness V3

Reproducible research project for developing and critically evaluating a
structure-based drug-likeness classification pipeline.

## Project status

Computational experiments are complete and the project is in the
publication-analysis phase.

The current pipeline includes:

- dataset construction and quality control
- molecular standardization and deduplication
- leakage auditing
- random and scaffold-aware evaluation
- Morgan fingerprint and descriptor representations
- Random Forest and logistic-regression baselines
- repeated scaffold evaluation
- bootstrap uncertainty estimation
- applicability-domain analysis
- systematic error analysis
- descriptor-calibrated sensitivity analyses
- external validation on ClinTox
- publication figure generation

The current computational results are frozen unless a reproducibility
audit identifies a methodological error.

## Scientific objective

The project evaluates how well structure-based molecular representations
discriminate the operational positive and negative classes defined in the
development dataset, with particular emphasis on whether apparent
performance remains after increasingly stringent validation procedures.

## Dataset

The final modeling dataset contains 8,196 unique standardized molecules:

| Source | Label 0 | Label 1 | Total |
|---|---:|---:|---:|
| ChEBI | 4,098 | 0 | 4,098 |
| DrugCentral | 0 | 4,098 | 4,098 |
| **Total** | **4,098** | **4,098** | **8,196** |

### Important limitation

The operational class label is perfectly confounded with source dataset:

- ChEBI → label 0
- DrugCentral → label 1

Therefore, the task must not be interpreted as a validated universal
"drug-likeness" endpoint.

The project explicitly investigates this limitation through:

- descriptor common-support analysis
- matched training sensitivity analysis
- stringent caliper matching
- applicability-domain analysis
- external validation

## Evaluation strategy

The primary evaluation hierarchy is:

1. random split
2. scaffold-separated split
3. repeated scaffold evaluation
4. descriptor-calibrated sensitivity analysis
5. external validation

The scaffold-aware analyses are treated as the more conservative estimates
of generalization.

## Main results

### Random split

Test performance:

- ROC-AUC: 0.9164
- PR-AUC: 0.9001
- Accuracy: 0.8488
- F1: 0.8588

### Repeated scaffold evaluation

Five scaffold-separated seeds produced:

- Random Forest ROC-AUC: 0.8924 ± 0.0219
- Random Forest PR-AUC: 0.8775 ± 0.0283
- Logistic regression ROC-AUC: 0.8518 ± 0.0187
- Logistic regression PR-AUC: 0.8454 ± 0.0158

### Descriptor-calibrated sensitivity analysis

The stringent caliper-matched analysis produced:

- ROC-AUC: 0.8745 ± 0.0227
- PR-AUC: 0.8674 ± 0.0254

Relative to the repeated scaffold analysis:

- paired ROC-AUC difference: −0.0178
- 95% CI: −0.0221 to −0.0136
- paired PR-AUC difference: −0.0100
- 95% CI: −0.0210 to 0.0010

These results indicate a modest reduction in performance after stronger
descriptor balancing rather than a complete loss of discrimination.

## Applicability domain

Performance varies substantially with molecular similarity to the training
set.

The applicability-domain analysis therefore forms an important part of
interpretation: performance should not be assumed to remain constant for
molecules distant from the development chemical space.

## External validation

An external ClinTox evaluation was performed after:

- invalid/query structure exclusion
- within-external deduplication
- development-set overlap removal

Final external set:

- 934 molecules
- 893 approved
- 41 non-approved

ClinTox ROC-AUC:

- 0.4701
- bootstrap 95% CI: 0.3871–0.5549

This result indicates limited transportability.

Importantly, ClinTox uses an FDA-approval endpoint rather than the
source-derived ChEBI-versus-DrugCentral development label. Therefore,
the external result should be interpreted as evidence concerning
cross-dataset/domain transportability rather than as a direct replication
of the original classification task.

## Reproducibility

The computational workflow is organized into reusable package code under
`src/` and executable analysis scripts under `scripts/`.

Generated tables and figures are intentionally excluded from version
control and can be regenerated from the analysis scripts.

The research manifest provides an experiment-to-output mapping:

`RESEARCH_MANIFEST.md`

Publication figures are generated using:

`scripts/plot_publication_figures.py`

Figures are written to:

`results/figures/`

## Repository structure

- `data/` — raw, external, intermediate, and processed datasets
- `src/` — reusable Python package
- `scripts/` — executable analysis and evaluation scripts
- `tests/` — automated tests
- `configs/` — project configuration
- `results/` — generated tables, figures, and model artifacts
- `RESEARCH_MANIFEST.md` — experiment and reproducibility manifest

## Scientific workflow

Dataset and endpoint definition  
→ data standardization  
→ deduplication  
→ labeling  
→ quality control  
→ leakage audit  
→ dataset splitting  
→ molecular representations  
→ baseline modeling  
→ scaffold-aware validation  
→ repeated robustness analysis  
→ uncertainty estimation  
→ applicability-domain analysis  
→ error analysis  
→ descriptor-calibrated sensitivity analysis  
→ external validation  
→ publication analysis

## Interpretation principle

High internal performance is not treated as sufficient evidence of
general-purpose drug-likeness prediction.

The manuscript should explicitly distinguish:

- performance on the constructed development classification task
- scaffold-based generalization
- robustness after descriptor balancing
- applicability-domain dependence
- external transportability

Limitations, particularly source-label confounding and endpoint mismatch
during external validation, are considered central scientific findings
rather than secondary implementation details.