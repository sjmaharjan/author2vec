from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def get_classifiers(name):
    classifiers = {
        "lr": (
            LogisticRegression(class_weight="balanced"),
            {"C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]},
        ),
        "svc": (
            LinearSVC(class_weight="balanced"),
            {
                "C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000],
                "class_weight": ["balanced", None],
            },
        ),
        "rf": (
            RandomForestClassifier(),
            {
                "n_estimators": [25, 100, 200, 300, 400, 500],
                "max_depth": [2, None],
                "bootstrap": [True, False],
                "criterion": ["gini", "entropy"],
            },
        ),
        "percepton": (
            Perceptron(n_iter=50),
            {
                "penalty": [None, "l2", "l1", "elasticnet"],
                "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
            },
        ),
        "adaboost": (
            AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
            {
                "n_estimators": [25, 100, 200, 300, 400, 500],
                "base_estimator__max_depth": [2, None],
                "base_estimator__criterion": ["gini", "entropy"],
            },
        ),
    }

    return classifiers.get(name, None)
