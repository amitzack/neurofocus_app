## -*- coding: utf-8 -*-
"""
app.py — NeuroFocus Assist backend

טוען את מודל ה-KNN המאומן (מ-model/) ומחזיר תחזית אמיתית
עבור חלון/חלונות EEG שמועלים דרך ה-frontend.

הרצה מקומית:  python app.py
פריסה ל-Render: gunicorn app:app
"""

import os
import io
import json
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from joblib import load
import scipy.io as sio

from features import raw_recording_to_features

app = Flask(__name__, static_folder="static", static_url_path="")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

# ---------------------------------------------------------
# טעינת המודל וקבצי העזר פעם אחת, בעליית השרת
# ---------------------------------------------------------
_clf = None
_selected_idx = None
_mean = None
_std = None
_meta = None
_load_error = None

def load_artifacts():
    global _clf, _selected_idx, _mean, _std, _meta, _load_error
    try:
        _clf = load(os.path.join(MODEL_DIR, "knn_model.joblib"))
        _selected_idx = np.load(os.path.join(MODEL_DIR, "selected_indices.npy"))
        _mean = np.load(os.path.join(MODEL_DIR, "feat_mean.npy"))
        _std = np.load(os.path.join(MODEL_DIR, "feat_std.npy"))
        with open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8") as f:
            _meta = json.load(f)
        print("Model artifacts loaded successfully.")
        print(f"  raw_feature_len = {_meta['raw_feature_len']}")
        print(f"  n_features_selected = {_meta['n_features_selected']}")
    except Exception as e:
        _load_error = str(e)
        print(f"WARNING: could not load model artifacts: {e}")
        print("The /api/predict endpoint will return an error until model files exist in ./model/")

load_artifacts()


def parse_uploaded_file(file_storage):
    """
    קורא קובץ הקלטת EEG גולמית שהועלה ומחזיר מערך numpy בצורת
    (n_samples, n_channels) — לפני כל עיבוד.

    נתמכים:
      .mat  — כמו קבצי ה-EEG הגולמיים במאגר (מפתח פנימי כלשהו, למשל v1p)
      .npy  — מערך numpy גולמי באותה צורה
      .csv  — טבלה גולמית באותה צורה
    """
    filename = file_storage.filename.lower()
    raw_bytes = file_storage.read()

    if filename.endswith(".mat"):
        mat = sio.loadmat(io.BytesIO(raw_bytes))
        data_keys = [k for k in mat.keys() if not k.startswith("__")]
        if not data_keys:
            raise ValueError("No data variable found inside the .mat file.")
        # לוקחים את המשתנה הראשון שבאמת נראה כמו מטריצת EEG (2 ממדים)
        arr = None
        for k in data_keys:
            candidate = np.asarray(mat[k])
            if candidate.ndim == 2 and min(candidate.shape) > 1:
                arr = candidate
                break
        if arr is None:
            raise ValueError(f"Could not find a valid EEG matrix among: {data_keys}")
    elif filename.endswith(".npy"):
        arr = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
    elif filename.endswith(".csv") or filename.endswith(".txt"):
        arr = np.loadtxt(io.BytesIO(raw_bytes), delimiter=",")
    else:
        raise ValueError("Unsupported file type. Please upload a .mat, .npy or .csv file.")

    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # אם זו הקלטת EEG גולמית שנטענה הפוך (19, samples) במקום (samples, 19)
    if arr.ndim == 2 and arr.shape[0] == 19 and arr.shape[1] != 19:
        arr = arr.T

    return arr


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    ok = _clf is not None
    return jsonify({
        "status": "ready" if ok else "model_not_loaded",
        "error": _load_error,
        "meta": _meta,
    }), (200 if ok else 503)


@app.route("/api/predict", methods=["POST"])
def predict():
    if _clf is None:
        return jsonify({
            "error": "Model is not loaded on the server yet. "
                     "Run export_model.py locally and upload the artifacts to /model."
        }), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Expected form field 'file'."}), 400

    file_storage = request.files["file"]
    if file_storage.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        arr = parse_uploaded_file(file_storage)
    except Exception as e:
        return jsonify({"error": f"Could not parse file: {e}"}), 400

    expected_len = _meta["raw_feature_len"]

    # זיהוי אוטומטי: האם זה כבר קובץ פיצ'רים מוכן (כמו features_subj{i}.npy,
    # n_windows x 1501), או הקלטת EEG גולמית (n_samples x 19) שצריך לעבד?
    if arr.ndim == 2 and arr.shape[1] == expected_len:
        # כבר פיצ'רים מוכנים — משתמשים ישירות
        X_raw = arr
    elif arr.ndim == 2 and arr.shape[1] == 19:
        # EEG גולמי — מריצים חילוץ מאפיינים
        try:
            X_raw = raw_recording_to_features(arr)
        except Exception as e:
            return jsonify({"error": f"Feature extraction failed: {e}"}), 400
    else:
        return jsonify({
            "error": f"Unrecognized file shape {arr.shape}. Expected either a "
                     f"raw EEG recording (samples x 19) or a precomputed "
                     f"feature file (windows x {expected_len})."
        }), 400

    # ניקוי NaN/Inf בסיסי
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # בחירת פיצ'רים + נרמול, בדיוק כמו באימון
    X_sel = X_raw[:, _selected_idx]
    X_norm = (X_sel - _mean) / _std

    preds = _clf.predict(X_norm)
    probs = _clf.predict_proba(X_norm)

    label_map = _meta["label_map"]  # {"0": "ADHD", "1": "Control"}
    n_windows = len(preds)

    votes_adhd = int(np.sum(preds == 0))
    votes_control = int(np.sum(preds == 1))

    majority_label = 0 if votes_adhd >= votes_control else 1
    # ביטחון = ממוצע ה-probability של המחלקה שזכתה, על פני החלונות שהצביעו לה
    class_col = majority_label
    relevant_probs = probs[preds == majority_label, class_col]
    confidence = float(np.mean(relevant_probs)) if len(relevant_probs) else float(np.max(probs))

    return jsonify({
        "label": label_map[str(majority_label)],
        "label_code": int(majority_label),
        "confidence_pct": round(confidence * 100, 1),
        "n_windows": n_windows,
        "votes": {"ADHD": votes_adhd, "Control": votes_control},
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
