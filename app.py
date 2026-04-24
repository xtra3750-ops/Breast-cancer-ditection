from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = Flask(__name__)

FEATURES = [
    {
        "key": "worst_concave_points",
        "label": "Enter number from report for worst concave points",
        "dataset_name": "worst concave points",
        "tip": "Type the exact value shown next to worst concave points in your report.",
    },
    {
        "key": "worst_texture",
        "label": "Enter number from report for worst texture",
        "dataset_name": "worst texture",
        "tip": "Type the exact value shown next to worst texture in your report.",
    },
    {
        "key": "area_error",
        "label": "Enter number from report for area error",
        "dataset_name": "area error",
        "tip": "Type the exact value shown next to area error in your report.",
    },
    {
        "key": "mean_texture",
        "label": "Enter number from report for mean texture",
        "dataset_name": "mean texture",
        "tip": "Type the exact value shown next to mean texture in your report.",
    },
    {
        "key": "mean_smoothness",
        "label": "Enter number from report for mean smoothness",
        "dataset_name": "mean smoothness",
        "tip": "Type the exact value shown next to mean smoothness in your report.",
    },
    {
        "key": "worst_perimeter",
        "label": "Enter number from report for worst perimeter",
        "dataset_name": "worst perimeter",
        "tip": "Type the exact value shown next to worst perimeter in your report.",
    },
    {
        "key": "compactness_error",
        "label": "Enter number from report for compactness error",
        "dataset_name": "compactness error",
        "tip": "Type the exact value shown next to compactness error in your report.",
    },
    {
        "key": "worst_radius",
        "label": "Enter number from report for worst radius",
        "dataset_name": "worst radius",
        "tip": "Type the exact value shown next to worst radius in your report.",
    },
    {
        "key": "mean_compactness",
        "label": "Enter number from report for mean compactness",
        "dataset_name": "mean compactness",
        "tip": "Type the exact value shown next to mean compactness in your report.",
    },
    {
        "key": "worst_symmetry",
        "label": "Enter number from report for worst symmetry",
        "dataset_name": "worst symmetry",
        "tip": "Type the exact value shown next to worst symmetry in your report.",
    },
]


def build_model():
    data = load_breast_cancer()
    feature_names = list(data.feature_names)
    indices = [feature_names.index(item["dataset_name"]) for item in FEATURES]
    X = data.data[:, indices]
    y = data.target

    model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=100,
        learning_rate=1.0,
        random_state=42,
    )
    model.fit(X, y)
    return model


MODEL = build_model()


@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    values = []

    for field in FEATURES:
        key = field["key"]
        value = payload.get(key)
        if value is None or str(value).strip() == "":
            return jsonify({"error": f"Please enter a value for {field['label']}"}), 400
        try:
            values.append(float(value))
        except ValueError:
            return jsonify({"error": f"{field['label']} must be a number."}), 400

    features = np.array(values).reshape(1, -1)
    prediction = MODEL.predict(features)[0]
    probability = float(np.max(MODEL.predict_proba(features)))

    if prediction == 1:
        result_text = "Benign — low risk"
    else:
        result_text = "Malignant — higher risk"

    return jsonify(
        {
            "prediction": result_text,
            "confidence": round(probability * 100, 1),
            "message": "This result is a prediction only. Please consult a doctor for a full diagnosis.",
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
