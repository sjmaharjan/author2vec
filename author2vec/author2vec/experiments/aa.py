import os
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from contextlib import redirect_stdout
from sklearn.model_selection import StratifiedKFold
import numpy as np


def aa_classification(books, labels, feature_extractor, features, out_file):
    with open(out_file, "w") as f:
        with redirect_stdout(f):
            print("Features {}".format(features))

            print("Total instances: {}".format(len(books)))
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(labels)

            print("Number of classes: {}".format(label_encoder.classes_))

            books = np.array(books)

            scores = []

            skf = StratifiedKFold(y=labels, n_folds=5, shuffle=True, random_state=1234)
            for train_index, test_index in skf:

                X_train, X_test = books[train_index], books[test_index]
                y_train, y_test = labels[train_index], labels[test_index]

                train_features = feature_extractor.fit_transform(X_train)
                test_features = feature_extractor.transform(X_test)

                clf = LinearSVC(random_state=1234, class_weight="balanced")

                param_grid = {"C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 100]}

                print("Param Grid {}".format(param_grid))
                print(
                    "Parameter of feature extractor  {}".format(
                        feature_extractor.get_params()
                    )
                )
                # Train a SVM classification model
                print("exp start")
                grid = GridSearchCV(
                    clf, param_grid, scoring="accuracy", n_jobs=3, refit=True
                )
                grid.fit(train_features, y_train)
                scores.append(grid.score(test_features, y_test))

            print("All folds accuracry {}".format(scores))
            print("Mean Accuracy: {}".format(np.mean(scores)))
            print("Std Accuracy: {}".format(np.std(scores)))
            print("Done")


def collect_results(fpath):
    import pandas as pd

    all_results = []
    found = False
    with open(fpath, "r") as f_in:
        results = {}
        count = 0
        for line in f_in.readlines():
            if line.startswith("==>"):
                method_name = line.strip("==>").strip("<==")
                found = True
                results["method"] = method_name

            if found and line.startswith("Mean Accuracy:"):
                results["mean"] = line.split(":")[1]
                count += 1

            if found and line.startswith("Std Accuracy:"):
                results["std"] = line.split(":")[1]
                count += 1

            if found and line.startswith("All folds accuracry"):
                results["folds"] = line.split("accuracry")[1]
                count += 1

            if count == 3:
                all_results.append(results)
                found == False
                results = {}
                count = 0

    df = pd.DataFrame(all_results)
    df.to_csv("aa_results.tsv", sep="\t")


if __name__ == "__main__":
    collect_results("/Users/sjmaharjan/author_style/aa_results/all_12_results.results")

