from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"

FIGURES.mkdir(
    parents=True,
    exist_ok=True,
)


def save_figure(fig, name):
    png_path = FIGURES / f"{name}.png"
    pdf_path = FIGURES / f"{name}.pdf"

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Created: {png_path}")
    print(f"Created: {pdf_path}")


def figure_1_evaluation_comparison():
    baseline = pd.read_csv(
        TABLES / "baseline_results.csv"
    )

    repeated = pd.read_csv(
        TABLES / "repeated_scaffold_results.csv"
    )

    matched = pd.read_csv(
        TABLES / "matched_sensitivity_comparison.csv"
    )

    random_roc = baseline.loc[
        baseline["split"] == "test",
        "roc_auc",
    ].iloc[0]

    random_pr = baseline.loc[
        baseline["split"] == "test",
        "pr_auc",
    ].iloc[0]

    scaffold_roc = repeated["rf_roc_auc"].mean()
    scaffold_roc_sd = repeated["rf_roc_auc"].std(ddof=1)

    scaffold_pr = repeated["rf_pr_auc"].mean()
    scaffold_pr_sd = repeated["rf_pr_auc"].std(ddof=1)

    matched_roc = matched["matched_roc_auc"].mean()
    matched_roc_sd = matched["matched_roc_auc"].std(ddof=1)

    matched_pr = matched["matched_pr_auc"].mean()
    matched_pr_sd = matched["matched_pr_auc"].std(ddof=1)

    labels = [
        "Random split",
        "Repeated scaffold",
        "Caliper-matched",
    ]

    roc_values = [
        random_roc,
        scaffold_roc,
        matched_roc,
    ]

    roc_errors = [
        0,
        scaffold_roc_sd,
        matched_roc_sd,
    ]

    pr_values = [
        random_pr,
        scaffold_pr,
        matched_pr,
    ]

    pr_errors = [
        0,
        scaffold_pr_sd,
        matched_pr_sd,
    ]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(
        figsize=(7.5, 5.5)
    )

    width = 0.36

    ax.bar(
        x - width / 2,
        roc_values,
        width,
        yerr=roc_errors,
        capsize=4,
        label="ROC-AUC",
    )

    ax.bar(
        x + width / 2,
        pr_values,
        width,
        yerr=pr_errors,
        capsize=4,
        label="PR-AUC",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=15,
        ha="right",
    )

    ax.set_ylabel("Performance")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Performance under increasingly stringent evaluation"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_1_evaluation_comparison",
    )


def figure_2_repeated_scaffold():
    results = pd.read_csv(
        TABLES / "repeated_scaffold_results.csv"
    )

    results = results.sort_values(
        "seed"
    )

    x = np.arange(
        len(results)
    )

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.plot(
        x,
        results["rf_roc_auc"],
        marker="o",
        linewidth=2,
        label="Random Forest",
    )

    ax.plot(
        x,
        results["lr_roc_auc"],
        marker="o",
        linewidth=2,
        label="Logistic regression",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        results["seed"].astype(str),
        rotation=30,
        ha="right",
    )

    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.75, 1.0)

    ax.set_title(
        "Repeated scaffold evaluation"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_2_repeated_scaffold",
    )


def figure_3_applicability_domain():
    data = pd.read_csv(
        TABLES / "applicability_domain_summary.csv"
    )

    order = [
        "<0.4",
        "0.4-0.5",
        "0.5-0.6",
        "0.6-0.7",
        "0.7-0.8",
        "0.8-0.9",
        ">=0.9",
    ]

    data["similarity_bin"] = pd.Categorical(
        data["similarity_bin"],
        categories=order,
        ordered=True,
    )

    data = data.sort_values(
        "similarity_bin"
    )

    x = np.arange(
        len(data)
    )

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.plot(
        x,
        data["roc_auc"],
        marker="o",
        linewidth=2,
        label="ROC-AUC",
    )

    ax.plot(
        x,
        data["pr_auc"],
        marker="o",
        linewidth=2,
        label="PR-AUC",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        data["similarity_bin"],
    )

    ax.set_xlabel(
        "Nearest training-molecule similarity"
    )

    ax.set_ylabel("Performance")
    ax.set_ylim(0.5, 1.05)

    ax.set_title(
        "Applicability-domain dependence"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_3_applicability_domain",
    )


def figure_4_descriptor_balance():
    data = pd.read_csv(
        TABLES / "caliper_matched_descriptor_balance.csv"
    )

    data = data.copy()

    data["abs_before"] = (
        data["smd_before"].abs()
    )

    data["abs_after"] = (
        data["smd_after"].abs()
    )

    data = data.sort_values(
        "abs_before",
        ascending=True,
    )

    y = np.arange(
        len(data)
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.scatter(
        data["smd_before"],
        y,
        s=55,
        label="Before matching",
    )

    ax.scatter(
        data["smd_after"],
        y,
        s=55,
        label="After matching",
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.axvline(
        -0.1,
        linestyle="--",
        linewidth=1,
    )

    ax.axvline(
        0.1,
        linestyle="--",
        linewidth=1,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        data["descriptor"]
    )

    ax.set_xlabel(
        "Standardized mean difference"
    )

    ax.set_title(
        "Descriptor balance before and after caliper matching"
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_4_descriptor_balance",
    )


def figure_5_external_validation():
    development = pd.read_csv(
        TABLES / "repeated_scaffold_results.csv"
    )

    external = pd.read_csv(
        TABLES / "clintox_external_bootstrap.csv"
    )

    development_roc = (
        development["rf_roc_auc"].mean()
    )

    development_sd = (
        development["rf_roc_auc"].std(ddof=1)
    )

    external_roc = external.loc[
        external["metric"] == "roc_auc",
        "point_estimate",
    ].iloc[0]

    external_lower = external.loc[
        external["metric"] == "roc_auc",
        "ci_lower",
    ].iloc[0]

    external_upper = external.loc[
        external["metric"] == "roc_auc",
        "ci_upper",
    ].iloc[0]

    labels = [
        "Repeated scaffold\n(development)",
        "ClinTox\n(external)",
    ]

    values = [
        development_roc,
        external_roc,
    ]

    errors = [
        development_sd,
        (
            external_roc - external_lower,
            external_upper - external_roc,
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5.5)
    )

    ax.bar(
        [0, 1],
        values,
        yerr=[
            development_sd,
            external_roc - external_lower,
        ],
        capsize=5,
    )

    # Add the asymmetric external upper interval
    ax.errorbar(
        1,
        external_roc,
        yerr=[
            [external_roc - external_lower],
            [external_upper - external_roc],
        ],
        fmt="none",
        capsize=5,
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "ROC-AUC"
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_title(
        "Development performance and external transportability"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_5_external_validation",
    )


def main():
    print(
        "=== Publication figure generation ==="
    )

    figure_1_evaluation_comparison()
    figure_2_repeated_scaffold()
    figure_3_applicability_domain()
    figure_4_descriptor_balance()
    figure_5_external_validation()

    print("\nOutput directory:")
    print(FIGURES)


if __name__ == "__main__":
    main()