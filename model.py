from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class SplitData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def _news_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("Top")]


def build_text(df: pd.DataFrame) -> pd.Series:
    cols = _news_columns(df)
    if not cols:
        raise ValueError("Nuk u gjetën kolonat e lajmeve (Top1..Top25).")
    # Join Top1..Top25 into one document per day
    # NOTE: On some pandas versions, DataFrame.agg with `" ".join` can return a DataFrame,
    # which then breaks `.str` operations. Using apply guarantees a 1D Series output.
    text = df[cols].fillna("").astype(str).apply(lambda r: " ".join(r.values), axis=1)
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def split_by_year(df: pd.DataFrame, split_year: int = 2015) -> SplitData:
    d = df.dropna(subset=["Date", "Label"]).copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["Year"] = d["Date"].dt.year
    train_df = d[d["Year"] < split_year].copy()
    test_df = d[d["Year"] >= split_year].copy()
    return SplitData(train_df=train_df, test_df=test_df)


def make_pipeline(max_features: int = 50000, ngram_max: int = 2, C: float = 2.0) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, ngram_max),
                    lowercase=True,
                    stop_words="english",
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, C=C, n_jobs=None)),
        ]
    )


def train_and_evaluate(
    df: pd.DataFrame,
    split_year: int = 2015,
    max_features: int = 50000,
    ngram_max: int = 2,
    C: float = 2.0,
) -> dict:
    s = split_by_year(df, split_year=split_year)
    X_train = build_text(s.train_df)
    y_train = s.train_df["Label"].astype(int).to_numpy()
    X_test = build_text(s.test_df)
    y_test = s.test_df["Label"].astype(int).to_numpy()

    pipe = make_pipeline(max_features=max_features, ngram_max=ngram_max, C=C)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    proba = None
    auc = np.nan
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)[:, 1]
        try:
            auc = float(roc_auc_score(y_test, proba))
        except Exception:
            auc = np.nan

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred)
    return {
        "model": pipe,
        "split_year": split_year,
        "n_train": int(len(s.train_df)),
        "n_test": int(len(s.test_df)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
        "report": report,
        "test_pred": pred,
        "test_proba": proba,
        "test_df": s.test_df.reset_index(drop=True),
    }

