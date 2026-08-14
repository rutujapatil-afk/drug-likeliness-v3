from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLES = PROJECT_ROOT / "results" / "tables"

OUTPUT = TABLES / "publication_evidence_matrix.csv"


def add_row(
    rows,
    section,
    claim,
    analysis,
    source_file,
    metric,
    estimate,
    lower_ci="",
    upper_ci="",
    interpretation="",
):
    rows.append(
        {
            "section": section,
            "claim": claim,
            "analysis": analysis,
            "source_file": source_file,
            "metric": metric,
            "estimate": estimate,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "interpretation": interpretation,
        }
    )


def main():
    rows = []

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "modeling_dataset.csv"
    )

    add_row(
        rows,
        "Dataset",
        "Final modeling dataset contains 8,196 molecules.",
        "Final modeling dataset",
        "data/processed/modeling_dataset.csv",
        "n",
        len(dataset),
        interpretation="Final standardized modeling dataset.",
    )

    source_counts = (
        dataset["source_dataset"]
        .value_counts()
        .sort_index()
    )

    for source, count in source_counts.items():
        add_row(
            rows,
            "Dataset",
            f"{source} contributes {count} molecules.",
            "Dataset composition",
            "data/processed/modeling_dataset.csv",
            "n",
            int(count),
        )

    # ---------------------------------------------------------
    # Random split baseline
    # ---------------------------------------------------------

    baseline = pd.read_csv(
        TABLES / "baseline_results.csv"
    )

    test_baseline = baseline.loc[
        baseline["split"] == "test"
    ].iloc[0]

    for metric in [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:
        add_row(
            rows,
            "Results",
            "Random-split test performance.",
            "Random-split Morgan Random Forest",
            "results/tables/baseline_results.csv",
            metric,
            float(test_baseline[metric]),
        )

    # ---------------------------------------------------------
    # Repeated scaffold
    # ---------------------------------------------------------

    repeated = pd.read_csv(
        TABLES / "repeated_scaffold_results.csv"
    )

    for metric in [
        "rf_roc_auc",
        "rf_pr_auc",
        "lr_roc_auc",
        "lr_pr_auc",
    ]:
        values = repeated[metric]

        mean = values.mean()
        sd = values.std(ddof=1)

        add_row(
            rows,
            "Robustness",
            "Repeated scaffold performance across five seeds.",
            "Repeated scaffold evaluation",
            "results/tables/repeated_scaffold_results.csv",
            metric,
            mean,
            interpretation=(
                f"Mean ± SD = {mean:.4f} ± {sd:.4f}"
            ),
        )

    # ---------------------------------------------------------
    # Bootstrap internal test
    # ---------------------------------------------------------

    bootstrap = pd.read_csv(
        TABLES / "bootstrap_test_metrics.csv"
    )

    for _, row in bootstrap.iterrows():
        add_row(
            rows,
            "Uncertainty",
            f"Bootstrap uncertainty for {row['metric']}.",
            "5,000-replicate bootstrap",
            "results/tables/bootstrap_test_metrics.csv",
            row["metric"],
            float(row["point_estimate"]),
            float(row["ci_lower"]),
            float(row["ci_upper"]),
            "Percentile bootstrap 95% CI.",
        )

    # ---------------------------------------------------------
    # Applicability domain
    # ---------------------------------------------------------

    applicability = pd.read_csv(
        TABLES / "applicability_domain_summary.csv"
    )

    for _, row in applicability.iterrows():
        add_row(
            rows,
            "Applicability domain",
            (
                "Performance varies across molecular "
                f"similarity bin {row['similarity_bin']}."
            ),
            "Nearest-training-molecule similarity analysis",
            "results/tables/applicability_domain_summary.csv",
            "roc_auc",
            float(row["roc_auc"]),
            interpretation=(
                f"n={int(row['n'])}; "
                f"mean similarity={row['mean_similarity']:.3f}"
            ),
        )

        add_row(
            rows,
            "Applicability domain",
            (
                "Performance varies across molecular "
                f"similarity bin {row['similarity_bin']}."
            ),
            "Nearest-training-molecule similarity analysis",
            "results/tables/applicability_domain_summary.csv",
            "pr_auc",
            float(row["pr_auc"]),
        )

    # ---------------------------------------------------------
    # Error analysis
    # ---------------------------------------------------------

    errors = pd.read_csv(
        TABLES / "error_group_statistics_corrected.csv"
    )

    for _, row in errors.iterrows():
        add_row(
            rows,
            "Error analysis",
            (
                f"{row['descriptor']} differs between "
                f"{row['group_a']} and {row['group_b']}."
            ),
            "Mann-Whitney U test with Benjamini-Hochberg FDR correction",
            "results/tables/error_group_statistics_corrected.csv",
            row["descriptor"],
            float(row["cliffs_delta"]),
            interpretation=(
                f"FDR-adjusted p={row['p_fdr_bh']:.4g}; "
                f"Cliff's delta={row['cliffs_delta']:.4f}"
            ),
        )

    # ---------------------------------------------------------
    # Descriptor overlap
    # ---------------------------------------------------------

    overlap = pd.read_csv(
        TABLES / "descriptor_overlap_analysis.csv"
    )

    for _, row in overlap.iterrows():
        add_row(
            rows,
            "Sensitivity",
            (
                f"{row['class']} common-support fraction "
                f"at threshold {row['threshold']}."
            ),
            "Descriptor common-support analysis",
            "results/tables/descriptor_overlap_analysis.csv",
            "fraction_within_threshold",
            float(
                row["fraction_within_threshold"]
            ),
            interpretation=(
                f"Median nearest distance="
                f"{row['median_nearest_distance']:.4f}; "
                f"mean nearest distance="
                f"{row['mean_nearest_distance']:.4f}"
            ),
        )

    # ---------------------------------------------------------
    # Caliper matched sensitivity
    # ---------------------------------------------------------

    matched = pd.read_csv(
        TABLES / "matched_sensitivity_comparison.csv"
    )

    for metric, column in [
        ("ROC-AUC", "delta_roc_auc"),
        ("PR-AUC", "delta_pr_auc"),
    ]:
        values = matched[column]

        mean = values.mean()
        sd = values.std(ddof=1)

        se = sd / (len(values) ** 0.5)

        # n = 5, df = 4
        t_critical = 2.776445

        lower = mean - t_critical * se
        upper = mean + t_critical * se

        add_row(
            rows,
            "Sensitivity",
            (
                f"Paired difference in {metric} after "
                "caliper-matched training."
            ),
            "Caliper-matched training sensitivity",
            "results/tables/matched_sensitivity_comparison.csv",
            f"delta_{metric.lower().replace('-', '_')}",
            mean,
            lower,
            upper,
            "Paired t-based 95% CI across five scaffold seeds.",
        )

    # ---------------------------------------------------------
    # Descriptor balance
    # ---------------------------------------------------------

    balance = pd.read_csv(
        TABLES / "caliper_matched_descriptor_balance.csv"
    )

    for _, row in balance.iterrows():
        add_row(
            rows,
            "Sensitivity",
            (
                f"Descriptor balance for {row['descriptor']} "
                "before and after caliper matching."
            ),
            "Caliper descriptor balance",
            "results/tables/caliper_matched_descriptor_balance.csv",
            "smd_after",
            float(row["smd_after"]),
            interpretation=(
                f"SMD before matching={row['smd_before']:.4f}"
            ),
        )

    # ---------------------------------------------------------
    # External validation
    # ---------------------------------------------------------

    external = pd.read_csv(
        TABLES / "clintox_external_bootstrap.csv"
    )

    for _, row in external.iterrows():
        add_row(
            rows,
            "External validation",
            (
                f"External ClinTox {row['metric']}."
            ),
            "ClinTox external validation with bootstrap uncertainty",
            "results/tables/clintox_external_bootstrap.csv",
            row["metric"],
            float(row["point_estimate"]),
            float(row["ci_lower"]),
            float(row["ci_upper"]),
            "5,000-replicate bootstrap 95% CI.",
        )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    evidence = pd.DataFrame(rows)

    evidence.to_csv(
        OUTPUT,
        index=False,
    )

    print(
        "=== Publication evidence matrix ==="
    )

    print(
        "Rows:",
        len(evidence),
    )

    print(
        "\nSections:"
    )

    print(
        evidence["section"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nOutput:"
    )

    print(OUTPUT)


if __name__ == "__main__":
    main()