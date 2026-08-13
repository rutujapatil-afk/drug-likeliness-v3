# Research Manifest

## Project

Drug-likeness molecular classification using structure-aware validation.

## Dataset

Final modeling dataset:

- File: `data/processed/modeling_dataset.csv`
- Molecules: 8,196
- ChEBI: 4,098
- DrugCentral: 4,098
- Classes: 4,098 negative / 4,098 positive
- Canonical SMILES duplicates: audited during dataset construction

## Critical labeling limitation

The operational labels are source-derived:

- ChEBI -> label 0
- DrugCentral -> label 1

Therefore source dataset and class label are perfectly confounded.

This limitation is explicitly addressed through descriptor common-support analysis and descriptor-calibrated sensitivity analyses.

---

## Core experiments

### Dataset characterization

Script:

`scripts/characterize_dataset.py`

Output:

`results/tables/dataset_characterization.csv`

### Random-split Morgan RF baseline

Script:

`scripts/train_baseline.py`

Output:

`results/tables/baseline_results.csv`

Test ROC-AUC: 0.9164

Test PR-AUC: 0.9001

### Descriptor RF baseline

Script:

`scripts/train_descriptor_baseline.py`

### Logistic regression baseline

Script:

`scripts/train_logistic_baseline.py`

### Scaffold-aware baseline

Script:

`scripts/train_scaffold_baseline.py`

### Descriptor scaffold model

Script:

`scripts/train_descriptor_scaffold.py`

### Combined Morgan + descriptor model

Script:

`scripts/train_combined_scaffold.py`

---

## Repeated scaffold evaluation

Script:

`scripts/repeat_scaffold_experiment.py`

Output:

`results/tables/repeated_scaffold_results.csv`

Five seeds:

- 20260813
- 20260814
- 20260815
- 20260816
- 20260817

Mean RF performance:

- ROC-AUC: 0.8924 ± 0.0219
- PR-AUC: 0.8775 ± 0.0283

---

## Statistical robustness

### Bootstrap test metrics

Script:

`scripts/bootstrap_test_metrics.py`

Output:

`results/tables/bootstrap_test_metrics.csv`

### Applicability domain

Script:

`scripts/analyze_applicability_domain.py`

Outputs:

- `results/tables/applicability_domain.csv`
- `results/tables/applicability_domain_summary.csv`

### Error analysis

Scripts:

- `scripts/error_analysis.py`
- `scripts/analyze_errors.py`
- `scripts/correct_error_statistics.py`

Outputs:

- `results/tables/scaffold_test_predictions.csv`
- `results/tables/error_group_statistics.csv`
- `results/tables/error_group_statistics_corrected.csv`

---

## Source-confounding sensitivity

### Descriptor common support

Script:

`scripts/analyze_descriptor_overlap.py`

Output:

`results/tables/descriptor_overlap_analysis.csv`

### Matched training sensitivity

Script:

`scripts/matched_training_sensitivity.py`

Outputs:

- `results/tables/matched_training_sensitivity.csv`
- `results/tables/matched_training_descriptor_balance.csv`

### Stringent caliper sensitivity

Script:

`scripts/caliper_matched_training_sensitivity.py`

Outputs:

- `results/tables/caliper_matched_training_results.csv`
- `results/tables/caliper_matched_descriptor_balance.csv`

### Paired sensitivity comparison

Script:

`scripts/compare_matched_sensitivity.py`

Output:

`results/tables/matched_sensitivity_comparison.csv`

Final caliper sensitivity:

- Original repeated scaffold ROC-AUC: 0.8924 ± 0.0219
- Caliper-matched ROC-AUC: 0.8745 ± 0.0227
- Paired mean difference: -0.0178
- 95% CI: -0.0221 to -0.0136

---

## External validation

Dataset:

ClinTox

Scripts:

- `scripts/validate_clintox_external.py`
- `scripts/bootstrap_clintox_external.py`

Outputs:

- `results/tables/clintox_external_predictions.csv`
- `results/tables/clintox_external_summary.csv`
- `results/tables/clintox_external_bootstrap.csv`

External ROC-AUC:

0.4701

Bootstrap 95% CI:

0.3871 to 0.5549

This result is interpreted as evidence of limited transportability across datasets/endpoints rather than hidden or discarded.

---

## Primary scientific conclusions

1. Morgan fingerprints provide strong discrimination in the constructed DrugCentral-versus-ChEBI classification task.
2. Scaffold-aware evaluation gives a more conservative estimate than random splitting.
3. Performance remains robust across repeated scaffold splits.
4. Performance persists after stringent descriptor-calibrated training sensitivity analysis.
5. Applicability-domain similarity strongly influences performance.
6. Error analysis identifies systematic physicochemical differences between false positives, false negatives, and correct predictions.
7. External ClinTox validation demonstrates limited cross-domain transportability.
8. The source-derived labeling scheme is a major limitation and must remain explicit in interpretation.

---

## Reproducibility principle

No additional model or hyperparameter search should be introduced solely to improve performance after the experimental freeze.

Further work should focus on:

- repository reproducibility
- publication figures
- manuscript tables
- statistical reporting
- transparent limitations
- manuscript preparation