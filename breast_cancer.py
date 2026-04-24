import numpy as np

try:
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    plt = None
    has_matplotlib = False

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def load_data():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return X, y, data


def build_model(random_state: int = 42):
    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=random_state)
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=100,
        learning_rate=1.0,
        random_state=random_state,
    )
    return model


def evaluate_model(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:\n")
    print(report)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return acc, report, cm


def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        plt.figure(figsize=(10, 6))
        plt.title("Top 15 Feature Importances")
        plt.barh(np.array(feature_names)[indices][::-1], importances[indices][::-1], color="tab:blue")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()
    else:
        print("The trained model does not expose feature_importances_.")


def main():
    X, y, data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = build_model(random_state=42)
    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test, data.target_names)
    plot_feature_importance(model, data.feature_names)


if __name__ == "__main__":
    main()