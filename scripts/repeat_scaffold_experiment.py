from pathlib import Path

import pandas as pd
from rdkit import RDLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from druglikeness.features import dataframe_to_morgan
from druglikeness.scaffold_split import scaffold_split


RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset.csv"
)

SEEDS = [
    20260813,
    20260814,
    20260815,
    20260816,
    20260817,
]


def evaluate(model, X, y):
    probability = model.predict_proba(X)[:, 1]

    return (
        roc_auc_score(y, probability),
        average_precision_score(y, probability),
    )


def main():
    dataframe = pd.read_csv(DATASET_PATH)

    results = []

    for seed in SEEDS:
        # scaffold_split currently uses its module-level seed.
        # We temporarily change it for this experiment.
        import druglikeness.scaffold_split as splitter

        splitter.RANDOM_SEED = seed

        train, validation, test = scaffold_split(dataframe)

        X_train = dataframe_to_morgan(train)
        X_test = dataframe_to_morgan(test)

        y_train = train["label"].to_numpy()
        y_test = test["label"].to_numpy()

        rf = RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )

        lr = LogisticRegression(
            max_iter=2000,
            random_state=seed,
            class_weight="balanced",
            solver="liblinear",
        )

        rf.fit(X_train, y_train)
        lr.fit(X_train, y_train)

        rf_auc, rf_pr = evaluate(
            rf,
            X_test,
            y_test,
        )

        lr_auc, lr_pr = evaluate(
            lr,
            X_test,
            y_test,
        )

        results.append(
            {
                "seed": seed,
                "train_size": len(train),
                "test_size": len(test),
                "rf_roc_auc": rf_auc,
                "rf_pr_auc": rf_pr,
                "lr_roc_auc": lr_auc,
                "lr_pr_auc": lr_pr,
            }
        )

        print(
            f"Seed {seed}: "
            f"RF ROC-AUC={rf_auc:.4f}, "
            f"RF PR-AUC={rf_pr:.4f}, "
            f"LR ROC-AUC={lr_auc:.4f}, "
            f"LR PR-AUC={lr_pr:.4f}"
        )

    results_df = pd.DataFrame(results)

    print("\n=== Summary ===")

    for column in [
        "rf_roc_auc",
        "rf_pr_auc",
        "lr_roc_auc",
        "lr_pr_auc",
    ]:
        print(
            f"{column}: "
            f"{results_df[column].mean():.4f} "
            f"+/- "
            f"{results_df[column].std(ddof=1):.4f}"
        )

    output_path = (
        PROJECT_ROOT
        / "results"
        / "tables"
        / "repeated_scaffold_results.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\nResults:", output_path)


if __name__ == "__main__":
    main()