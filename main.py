import argparse
from pprint import pprint
from typing import Dict, Final, Tuple

import joblib
import pandas as pd
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

CLASS_LBL: Final = "MachineLearning"
train_df_pth: Final = "data/train.csv.zip"
test_df_pth: Final = "data/test.csv.zip"


def feature_engineering(raw_df: DataFrame) -> DataFrame:
    df = raw_df.copy()
    df["CreationDate"] = pd.to_datetime(df["CreationDate"])
    df["CreationDate_Epoch"] = df["CreationDate"].astype("int64") // (10 ** 9)
    df[CLASS_LBL] = df["Tags"].str.contains("machine-learning").fillna(False)
    df = df.drop(columns=["Id", "Tags"])
    df["Title_Len"] = df["Title"].str.len()
    df["Body_Len"] = df["Body"].str.len()
    df = df.drop(columns=["FavoriteCount"])
    df["Text"] = df["Title"].fillna("") + " " + df["Body"].fillna("")

    return df


def fit_tfidf(
    train_df: DataFrame,
    test_df: DataFrame,
) -> Tuple[csr_matrix, csr_matrix, "TfidfVectorizer"]:
    tfidf = TfidfVectorizer(max_features=25000)
    train_tfidf = tfidf.fit_transform(train_df["Text"])
    test_tfidf = tfidf.transform(test_df["Text"])

    return train_tfidf, test_tfidf, tfidf


def fit_model(
    train_X: csr_matrix, train_y: Series, random_state=13
) -> LogisticRegression:
    log_clf_tfidf = SGDClassifier(loss="modified_huber", random_state=random_state)
    log_clf_tfidf.fit(train_X, train_y)

    return log_clf_tfidf


def eval_model(clf: LogisticRegression, X: csr_matrix, y: Series) -> Dict[str, float]:
    y_proba = clf.predict_proba(X)[:, 1]
    y_pred = clf.predict(X)

    return {
        "roc_auc": roc_auc_score(y, y_proba),
        "avg_precision": average_precision_score(y, y_proba),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }


def prepare_data(random_state=42):
    print("Loading data...")
    df = pd.read_csv("data/crossValidated-questions.csv")
    df[CLASS_LBL] = df["Tags"].str.contains("machine-learning").fillna(False)

    train_df: DataFrame
    test_df: DataFrame
    train_df, test_df = train_test_split(
        df, stratify=df[CLASS_LBL], random_state=random_state
    )

    print("Saving split data...")
    train_df.to_csv(train_df_pth)
    test_df.to_csv(test_df_pth)


def train():
    print("Loading data...")
    train_df = pd.read_csv(train_df_pth)
    test_df = pd.read_csv(test_df_pth)

    print("Engineering features...")
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)

    print("Fitting Tf-Idf...")
    train_tfidf, test_tfidf, tfidf = fit_tfidf(train_df, test_df)

    print("Saving Tf-Idf obj...")
    joblib.dump(tfidf, "artifacts/tfidf.jlib")

    print("Training model...")
    train_y = train_df[CLASS_LBL]
    model = fit_model(train_tfidf, train_y)

    print("Saving trained model...")
    joblib.dump(model, "artifacts/model.jlib")

    print("Evaluating model...")
    train_metrics = eval_model(model, train_tfidf, train_y)
    print("Train metrics:")
    pprint(train_metrics)
    print("-" * 10)

    test_metrics = eval_model(model, test_tfidf, test_df[CLASS_LBL])
    print("Test metrics:")
    pprint(test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="Prepare or Train step", dest="step")
    subparsers.required = True

    split_parser = subparsers.add_parser("prepare")
    split_parser.set_defaults(func=prepare_data)

    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train)

    parser.parse_args().func()
