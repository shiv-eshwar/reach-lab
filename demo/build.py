"""Export a stand-alone study replay and a portable three-feature research model."""
from pathlib import Path
from forest import explain_study
from offline import export_replay
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "demo"
frames = pd.read_pickle(ROOT / "processed/clean_frames.pkl")
features = pd.read_csv(ROOT / "processed/rep_features.csv", dtype={"subject": str})
if len(features) != 408 or features.position_error_mm.notna().sum() != 400:
    raise ValueError("Rebuild the corrected repetition features first.")
features["relative_deviation"] = features.max_lateral_deviation_mm / features.reach_distance_mm
cases, matrix, labels = [], [], []
for subject, rows in features.groupby("subject", sort=True):
    rows = rows.sort_values("rep")
    assert rows.rep.tolist() == list(range(1, 9))
    summary = [rows.path_efficiency.mean(), rows.path_efficiency.std(ddof=0), rows.relative_deviation.mean()]
    reps = []
    for rep, g in frames[frames.subject == subject].groupby("rep"):
        g = g.sort_values("frame")
        coords = g[[part + "_" + axis for part in ("shoulder","elbow","wrist","mcp3","object") for axis in ("x","y")]]
        points = coords.round(2).astype(object).where(coords.notna(), None).values.tolist()
        row = rows.loc[rows.rep == rep].iloc[0]
        reps.append(dict(points=points, efficiency=row.path_efficiency, duration=row.movement_duration_s,
                         endpoint=None if pd.isna(row.position_error_mm) else row.position_error_mm))
    cases.append(dict(id=subject, group=rows.iloc[0].group, side=rows.iloc[0].side, reps=reps, values=summary))
    matrix.append(summary)
    labels.append(int(rows.iloc[0].group=="DCP"))
X, y = np.asarray(matrix), np.asarray(labels)
names = ["Mean path efficiency", "Efficiency variation", "Mean relative deviation"]
def fit(indices):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=42)).fit(X[indices], y[indices])
    scale, classifier = pipe.steps[0][1], pipe.steps[1][1]
    return dict(mean=scale.mean_.tolist(), scale=scale.scale_.tolist(), weights=classifier.coef_[0].tolist(),
                intercept=classifier.intercept_[0], features=names)
def score(model, values):
    terms = (values - model["mean"]) / model["scale"] * model["weights"]
    return float(1 / (1 + np.exp(-(model["intercept"] + terms.sum())))), terms.tolist()
folds = []
cm = [[0,0],[0,0]]
for fold,(train,test) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(X,y,[c["id"] for c in cases]),1):
    model=fit(train)
    folds.append(dict(fold=fold,training=[cases[i]["id"] for i in train],test=[cases[i]["id"] for i in test],model=model))
    for i in test:
        prob, terms=score(model,X[i])
        cases[i].update(score=prob,terms=terms,fold=fold)
        cm[y[i]][int(prob>=0.5)]+=1
forest=explain_study(ROOT)
for case in cases:
    case["forest"]=forest[case["id"]]
result=dict(cases=cases, cm=cm, folds=folds, model=fit(np.arange(len(y))), fps=25,
            feature_sha256=hashlib.sha256((ROOT/"processed/rep_features.csv").read_bytes()).hexdigest())
path=out/"study.json"
tmp=path.with_suffix(".tmp")
tmp.write_text(json.dumps(result,separators=(",",":"),allow_nan=False))
tmp.replace(path)
export_replay()
print("Held-out confusion matrix (TD, DCP)",cm)
