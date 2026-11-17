"""Recover the exact benchmark fold forest and decompose its probability along tree paths."""
import ast
from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def explain_study(root):
    source = root / "results/corrected_benchmark"
    frame = pd.read_csv(root / "processed/experiment_dataset.csv", dtype={"subject": str}).set_index("subject")
    scores = pd.read_csv(source / "model_benchmark_predictions.csv", dtype={"subject": str}).query("model == 'Random_Forest'")
    config = pd.read_csv(source / "model_benchmark_fold_results.csv").query("model == 'Random_Forest'").set_index("fold")
    X = frame.drop(columns=["group","side","target"])
    out = {}
    for fold, held in scores.groupby("fold"):
        ids = held.subject.str.zfill(2).to_list()
        train = X.index.difference(ids, sort=False)
        model = Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),
                          ("selector",SelectKBest(partial(mutual_info_classif,random_state=42))),
                          ("model",RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=1))])
        model.set_params(**ast.literal_eval(config.loc[fold,"best_parameters"]))
        model.fit(X.loc[train],frame.loc[train,"target"])
        values = model[:-1].transform(X.loc[ids])
        names = X.columns[model.named_steps["selector"].get_support()]
        rf = model.named_steps["model"]
        probabilities = rf.predict_proba(values)[:,1]
        np.testing.assert_allclose(probabilities,held.predicted_score,atol=1e-10)
        for i,subject in enumerate(ids):
            terms=np.zeros(len(names))
            base=0
            for tree in rf.estimators_:
                t=tree.tree_
                probs=t.value[:,0,1]/t.value[:,0,:].sum(axis=1)
                node=0
                base+=probs[0]
                while t.children_left[node]!=t.children_right[node]:
                    feature=t.feature[node]
                    child=t.children_left[node] if np.float32(values[i,feature])<=t.threshold[node] else t.children_right[node]
                    terms[feature]+=probs[child]-probs[node]
                    node=child
            terms/=len(rf.estimators_)
            base/=len(rf.estimators_)
            assert np.isclose(base+terms.sum(),probabilities[i])
            order=np.argsort(-np.abs(terms))
            out[subject]=dict(score=probabilities[i],fold=int(fold),base=base,
                              terms=[[names[j],round(float(terms[j]),8)] for j in order],
                              call=int(held.iloc[i].predicted_target))
    return out
