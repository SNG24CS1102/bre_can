import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# --------------------------------------------------
# 1. Set random seed and Seaborn theme
# --------------------------------------------------

np.random.seed(42)
sns.set_theme(style="whitegrid")


# --------------------------------------------------
# 2. Load Breast Cancer Dataset
# --------------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

print("Dataset Shape:", X.shape)

print(
    f"Class distribution: {np.bincount(y)} "
    "(0: Malignant, 1: Benign)"
)


# --------------------------------------------------
# 3. Split Dataset into Training and Testing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 4. Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("Preprocessing complete.")


# --------------------------------------------------
# 5. MLE Logistic Regression
# --------------------------------------------------

mle_model = LogisticRegression(
    penalty=None,
    max_iter=10000
)

mle_model.fit(
    X_train_scaled,
    y_train
)


# --------------------------------------------------
# 6. MAP with L2 Regularization
# --------------------------------------------------

map_l2_model = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="lbfgs",
    max_iter=10000
)

map_l2_model.fit(
    X_train_scaled,
    y_train
)


# --------------------------------------------------
# 7. MAP with L1 Regularization
# --------------------------------------------------

map_l1_model = LogisticRegression(
    penalty="l1",
    solver="saga",
    C=1.0,
    max_iter=10000
)

map_l1_model.fit(
    X_train_scaled,
    y_train
)


# --------------------------------------------------
# 8. Compare Model Weights
# --------------------------------------------------

weights_df = pd.DataFrame({

    "Feature": ["Intercept"] + list(data.feature_names),

    "MLE": np.insert(
        mle_model.coef_[0],
        0,
        mle_model.intercept_[0]
    ),

    "MAP_L2 (Gaussian)": np.insert(
        map_l2_model.coef_[0],
        0,
        map_l2_model.intercept_[0]
    ),

    "MAP_L1 (Laplace)": np.insert(
        map_l1_model.coef_[0],
        0,
        map_l1_model.intercept_[0]
    )
})

print("\nFirst 10 Feature Weights:")
print(weights_df.head(10))


# --------------------------------------------------
# 9. Evaluation Function
# --------------------------------------------------

def evaluate(model, X, y, name):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(y, predictions),
        "Recall": recall_score(y, predictions),
        "F1-Score": f1_score(y, predictions),
        "AUC-ROC": roc_auc_score(y, probabilities)
    }


# --------------------------------------------------
# 10. Evaluate All Models
# --------------------------------------------------

results = [

    evaluate(
        mle_model,
        X_test_scaled,
        y_test,
        "MLE (No Regularization)"
    ),

    evaluate(
        map_l2_model,
        X_test_scaled,
        y_test,
        "MAP (L2 Regularization)"
    ),

    evaluate(
        map_l1_model,
        X_test_scaled,
        y_test,
        "MAP (L1 Regularization)"
    )
]

results_df = pd.DataFrame(results)

print("\nModel Evaluation Results:")
print(results_df.to_string(index=False))


# --------------------------------------------------
# 11. Plot Feature Weight Comparison
# --------------------------------------------------

plt.figure(figsize=(15, 6))

melted_w = weights_df.melt(
    id_vars="Feature",
    value_vars=[
        "MLE",
        "MAP_L2 (Gaussian)",
        "MAP_L1 (Laplace)"
    ],
    var_name="Method",
    value_name="Weight"
)

# Remove Intercept from graph
melted_w = melted_w[
    melted_w["Feature"] != "Intercept"
]

sns.barplot(
    data=melted_w,
    x="Feature",
    y="Weight",
    hue="Method"
)

plt.xticks(rotation=90)

plt.title("Parameter Estimate Comparisons")

plt.xlabel("Features")

plt.ylabel("Weight")

plt.tight_layout()


# --------------------------------------------------
# 12. Save Graph
# --------------------------------------------------

plt.savefig(
    "parameter_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nGraph saved as: parameter_comparison.png")