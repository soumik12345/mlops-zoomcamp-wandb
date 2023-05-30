"""@wandbcode{mlops-zoomcamp}
"""

import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import wandb
import pandas as pd


def run_train():
    # Initialize a WandB Run
    wandb.init()

    # Get hyperparameters from the run configs
    config = wandb.config

    # Fetch the latest version of the dataset artifact 
    artifact = wandb.use_artifact('geekyrakshit/mlops-zoomcamp-wandb/Titanic:latest', type='dataset')
    artifact_dir = artifact.download()

    # Read the files
    train_val_df = pd.read_csv(os.path.join(artifact_dir, "train.csv"))

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X_train = pd.get_dummies(train_val_df[features][train_val_df["Split"] == "Train"])
    X_val = pd.get_dummies(train_val_df[features][train_val_df["Split"] == "Validation"])
    y_train = train_val_df["Survived"][train_val_df["Split"] == "Train"]
    y_val = train_val_df["Survived"][train_val_df["Split"] == "Validation"]

    # Define and Train RandomForestClassifier model
    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        bootstrap=config.bootstrap,
        warm_start=config.warm_start,
        class_weight=config.class_weight,
    )
    model.fit(X_train, y_train)

    # Make Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_probas_val = model.predict_proba(X_val)

    # Log Metrics to Weights & Biases
    wandb.log({
        "Train/Accuracy": accuracy_score(y_train, y_pred_train),
        "Validation/Accuracy": accuracy_score(y_val, y_pred_val),
        "Train/Presicion": precision_score(y_train, y_pred_train),
        "Validation/Presicion": precision_score(y_val, y_pred_val),
        "Train/Recall": recall_score(y_train, y_pred_train),
        "Validation/Recall": recall_score(y_val, y_pred_val),
        "Train/F1-Score": f1_score(y_train, y_pred_train),
        "Validation/F1-Score": f1_score(y_val, y_pred_val),
    })

    # Plot plots to Weights & Biases
    label_names = ["Not-Survived", "Survived"]
    wandb.sklearn.plot_class_proportions(y_train, y_val, label_names)
    wandb.sklearn.plot_summary_metrics(model, X_train, y_train, X_val, y_val)
    wandb.sklearn.plot_roc(y_val, y_probas_val, labels=label_names)
    wandb.sklearn.plot_precision_recall(y_val, y_probas_val, labels=label_names)
    wandb.sklearn.plot_confusion_matrix(y_val, y_pred_val, labels=label_names)

    # Save your model
    with open("random_forest_classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    # Log your model as a versioned file to Weights & Biases Artifact
    artifact = wandb.Artifact("titanic-random-forest-model", type="model")
    artifact.add_file("random_forest_classifier.pkl")
    wandb.log_artifact(artifact)


SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "Validation/Accuracy", "goal": "maximize"},
    "parameters": {
        "max_depth": {
            "distribution": "int_uniform",
            "min": 1,
            "max": 20,
        },
        "n_estimators": {
            "distribution": "int_uniform",
            "min": 10,
            "max": 100,
        },
        "min_samples_split": {
            "distribution": "int_uniform",
            "min": 2,
            "max": 10,
        },
        "min_samples_leaf": {
            "distribution": "int_uniform",
            "min": 1,
            "max": 4,
        },
        "bootstrap": {"values": [True, False]},
        "warm_start": {"values": [True, False]},
        "class_weight": {"values": ["balanced", "balanced_subsample"]},
    },
}


if __name__ == "__main__":
    sweep_id = wandb.sweep(SWEEP_CONFIG, project="mlops-zoomcamp-wandb")
    wandb.agent(sweep_id, run_train, count=5)
