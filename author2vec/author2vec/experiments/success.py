from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import explained_variance_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
from contextlib import redirect_stdout
from author2vec.experiments import get_classifiers
from author2vec.experiments.utils import (
    output_fname_generator,
    print_classification_report,
)
from fauthor2vec.eatures.utils import fetch_features_vectorized
import os
from sklearn.preprocessing import normalize
from manage import app
import numpy as np
import tqdm


def balanced_accuracy(Y_actual, Y_predicted):

    return recall_score(Y_actual, Y_predicted, pos_label=None, average="macro")


def print_predictions(Y_test, Y_predictions):
    print("*" * 25)
    print("Actual, Predicted")
    for x, y in zip(Y_test.flatten(), np.array(Y_predictions).flatten()):
        print("{}, {}".format(x, y))

    print("*" * 25)


def get_prediction(corpus, feature, dump_dir):
    X_train, Y_train, X_test, Y_test = fetch_features_vectorized(
        dump_dir, feature, corpus
    )
    print("Running Experiment for Feature: {}".format(feature))
    print("============================================================")
    print("Training size: {}".format(len(Y_train)))
    print("Test size: {}".format(len(Y_test)))
    print("X Train Feature matrix {}".format(X_train.shape))
    print("X Test Feature matrix {}".format(X_test.shape))

    # svm = LinearSVC(random_state=1234)
    svm = LinearSVC(class_weight="balanced")

    param_grid = {"C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]}

    # Train a SVM classification model

    print("Fitting the classifier to the training set")

    grid = GridSearchCV(svm, param_grid, scoring="f1_weighted")
    grid.fit(X_train, Y_train)

    print("Best score: %0.3f" % grid.best_score_)
    print("Best parameters set:")
    best_parameters = grid.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    for params, mean_score, scores in grid.grid_scores_:
        print("%0.3f+/-%0.2f %r" % (mean_score, scores.std() / 2, params))

    print("Done grid search")
    print("============================================================")

    print("Predicting on the test set")

    y_train_pred = grid.predict(X_train)

    y_pred = grid.predict(X_test)

    target_names = ["failure", "success"]
    class_indices = {"failure": 0, "success": 1}
    train_acc = accuracy_score(Y_train, y_train_pred) * 100
    test_acc = accuracy_score(Y_test, y_pred) * 100

    print("Classifation Report")
    print("Training Accuracy ={}".format(train_acc))
    print("Test Accuracy ={}".format(test_acc))

    print(
        classification_report(
            Y_test,
            y_pred,
            target_names=target_names,
            labels=[class_indices[cls] for cls in target_names],
        )
    )

    y_pred_correct = [
        int(pred == actual)
        for pred, actual in zip(y_pred, [class_indices[cls] for cls in target_names])
    ]

    return y_pred, y_pred_correct


def success_classification(corpus, feature, dump_dir, ignore_lst):
    """

    :param corpus: Dataset
    :param feature:
    :param dump_dir:
    :return:
    """

    # classifiers=['lr','rf','percepton','svc']
    classifiers = ["svc"]

    for classifier in tqdm.tqdm(classifiers):
        print(classifier)

        clf, param_grid = get_classifiers(classifier)

        out_file = output_fname_generator(feature)
        out_file = "{}_{}.n.txt".format(out_file, classifier)
        with open(os.path.join(app.SUCCESS_OUTPUT, out_file), "w") as f:
            with redirect_stdout(f):

                X_train, Y_train, X_test, Y_test = fetch_features_vectorized(
                    dump_dir, feature, corpus
                )

                print("Running Experiment for Feature: {}".format(feature))
                print("============================================================")
                print("Training size: {}".format(len(Y_train)))
                print("Test size: {}".format(len(Y_test)))
                print("X Train Feature matrix {}".format(X_train.shape))
                print("X Test Feature matrix {}".format(X_test.shape))

                # svm = LinearSVC(random_state=1234)
                # svm = LinearSVC()

                # param_grid = {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]}

                # Train a SVM classification model

                print("Fitting the classifier to the training set")

                grid = GridSearchCV(clf, param_grid, scoring="f1_weighted")
                grid.fit(X_train, Y_train)

                print("Best score: %0.3f" % grid.best_score_)
                print("Best parameters set:")
                best_parameters = grid.best_estimator_.get_params()
                for param_name in sorted(param_grid.keys()):
                    print("\t%s: %r" % (param_name, best_parameters[param_name]))

                for params, mean_score, scores in grid.grid_scores_:
                    print("%0.3f+/-%0.2f %r" % (mean_score, scores.std() / 2, params))

                print("Done grid search")
                print("============================================================")

                print("Predicting on the test set")

                y_train_pred = grid.predict(X_train)

                y_pred = grid.predict(X_test)

                target_names = ["failure", "success"]
                class_indices = {"failure": 0, "success": 1}
                train_acc = accuracy_score(Y_train, y_train_pred) * 100
                test_acc = accuracy_score(Y_test, y_pred) * 100

                print("Classifation Report")
                print("Training Accuracy ={}".format(train_acc))
                print("Test Accuracy ={}".format(test_acc))

                print(
                    classification_report(
                        Y_test,
                        y_pred,
                        target_names=target_names,
                        labels=[class_indices[cls] for cls in target_names],
                    )
                )

                print("")
                print("Confusion matrix")
                print("============================================================")
                print(target_names)
                print()
                print(
                    confusion_matrix(
                        Y_test,
                        y_pred,
                        labels=[class_indices[cls] for cls in target_names],
                    )
                )

                p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
                    Y_test, y_pred, average="micro", pos_label=None
                )
                p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
                    Y_test, y_pred, average="macro", pos_label=None
                )
                p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
                    Y_test, y_pred, average="weighted", pos_label=None
                )
                roc_auc = roc_auc_score(y_true=Y_test, y_score=y_pred)

                # Precision recall f1 score for each class : success  and failure
                # For success
                precision_success = precision_score(
                    y_true=Y_test, y_pred=y_pred, average="binary", pos_label=1
                )
                recall_success = recall_score(
                    y_true=Y_test, y_pred=y_pred, average="binary", pos_label=1
                )
                f1_success = f1_score(
                    y_true=Y_test, y_pred=y_pred, average="binary", pos_label=1
                )

                # For Failure
                precision_failure = precision_score(
                    y_true=Y_test, y_pred=y_pred, average="binary", pos_label=0
                )
                recall_failure = recall_score(
                    y_true=Y_test, y_pred=y_pred, average="binary", pos_label=0
                )
                f1_failure = f1_score(
                    y_true=Y_test, y_pred=y_pred, average="binary", pos_label=0
                )
                bac = balanced_accuracy(Y_test, y_pred)

                print_predictions(Y_test, y_pred)

                print("")
                print(
                    "Accuracy,BAC,ROC_AUC,Precision-Macro,Precision-Micro,Precision-Weighted,Recall-Macro,Recall-Micro,Recall-Weighted,F1-Macro,F1-Micro,F1-Weighted,Failure-Precision,Failure-Recall,Failure-f1,Success-Precision,Success-Recall,Success-F1"
                )
                print(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
                        test_acc,
                        bac,
                        roc_auc,
                        p_macro,
                        p_micro,
                        p_weighted,
                        r_macro,
                        r_micro,
                        r_weighted,
                        f_macro,
                        f_micro,
                        f_weighted,
                        precision_failure,
                        recall_failure,
                        f1_failure,
                        precision_success,
                        recall_success,
                        f1_success,
                    )
                )
                print(
                    "Feature,{}, weighted F1,{}".format(
                        "::".join(feature) if isinstance(feature, list) else feature,
                        f_weighted,
                    )
                )


def success_classification_multitask(corpus, feature, dump_dir, ignore_lst):
    """

    :param corpus: Dataset
    :param feature:
    :param dump_dir:
    :return:
    """

    def transform_labels(Y):
        return np.array([int(label[-1]) for label in Y])

    # classifiers=['lr','rf','percepton','svc']
    classifiers = ["svc"]

    for classifier in tqdm.tqdm(classifiers):
        print(classifier)

        clf, param_grid = get_classifiers(classifier)

        out_file = output_fname_generator(feature)
        out_file = "{}_{}_n.txt".format(out_file, classifier)
        with open(os.path.join(app.SUCCESS_OUTPUT_MT, out_file), "w") as f:

            with redirect_stdout(f):

                X_train, Y_train, X_test, Y_test = fetch_features_vectorized(
                    dump_dir, feature, corpus
                )

                print("Running Experiment for Feature: {}".format(feature))
                print("============================================================")
                print("Training size: {}".format(len(Y_train)))
                print("Test size: {}".format(len(Y_test)))
                print("X Train Feature matrix {}".format(X_train.shape))
                print("X Test Feature matrix {}".format(X_test.shape))

                # svm = LinearSVC(random_state=1234)
                # svm = LinearSVC(random_state=1234)

                # param_grid = {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]}

                # Train a SVM classification model

                print("Fitting the classifier to the training set")

                grid = GridSearchCV(clf, param_grid, scoring="f1_weighted")
                grid.fit(X_train, Y_train)

                print("Best score: %0.3f" % grid.best_score_)
                print("Best parameters set:")
                best_parameters = grid.best_estimator_.get_params()
                for param_name in sorted(param_grid.keys()):
                    print("\t%s: %r" % (param_name, best_parameters[param_name]))

                for params, mean_score, scores in grid.grid_scores_:
                    print("%0.3f+/-%0.2f %r" % (mean_score, scores.std() / 2, params))

                print("Done grid search")
                print("============================================================")

                print("Predicting on the test set")

                y_train_pred = grid.predict(X_train)

                y_pred = grid.predict(X_test)

                target_names = np.unique(Y_train).tolist()
                # print (target_names)
                class_indices = {label: i for i, label in enumerate(target_names)}
                # print (class_indices)
                train_acc = accuracy_score(Y_train, y_train_pred) * 100
                test_acc = accuracy_score(Y_test, y_pred) * 100

                print("Classifation Report")
                print("Training Accuracy ={}".format(train_acc))
                print("Test Accuracy ={}".format(test_acc))

                print(classification_report(Y_test, y_pred, target_names=target_names))

                print("")
                print("Confusion matrix")
                print("============================================================")
                print(target_names)
                print()
                print(confusion_matrix(Y_test, y_pred))

                p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
                    Y_test, y_pred, average="micro", pos_label=None
                )
                p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
                    Y_test, y_pred, average="macro", pos_label=None
                )
                p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
                    Y_test, y_pred, average="weighted", pos_label=None
                )
                # roc_auc = roc_auc_score(y_true=Y_test, y_score=y_pred)

                print("")
                print(
                    "Accuracy,Precision-Macro,Precision-Micro,Precision-Weighted,Recall-Macro,Recall-Micro,Recall-Weighted,F1-Macro,F1-Micro,F1-Weighted"
                )
                print(
                    "{},{},{},{},{},{},{},{},{},{}".format(
                        test_acc,
                        p_macro,
                        p_micro,
                        p_weighted,
                        r_macro,
                        r_micro,
                        r_weighted,
                        f_macro,
                        f_micro,
                        f_weighted,
                    )
                )

                # print result for success class only

                print("============================================================")
                print("")
                print("")

                target_names = ["failure", "success"]
                class_indices = {"failure": 0, "success": 1}

                Y_train = transform_labels(Y_train)
                y_train_pred = transform_labels(y_train_pred)
                Y_test = transform_labels(Y_test)
                y_pred = transform_labels(y_pred)

                train_acc = accuracy_score(Y_train, y_train_pred) * 100
                test_acc = accuracy_score(Y_test, y_pred) * 100

                print("Classifation Report")
                print("Training Accuracy ={}".format(train_acc))
                print("Test Accuracy ={}".format(test_acc))

                print(
                    classification_report(
                        Y_test,
                        y_pred,
                        target_names=target_names,
                        labels=[class_indices[cls] for cls in target_names],
                    )
                )

                print("")
                print("Confusion matrix")
                print("============================================================")
                print(target_names)
                print()
                print(
                    confusion_matrix(
                        Y_test,
                        y_pred,
                        labels=[class_indices[cls] for cls in target_names],
                    )
                )

                p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
                    Y_test, y_pred, average="micro", pos_label=None
                )
                p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
                    Y_test, y_pred, average="macro", pos_label=None
                )
                p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
                    Y_test, y_pred, average="weighted", pos_label=None
                )
                roc_auc = roc_auc_score(y_true=Y_test, y_score=y_pred)

                # Precision recall f1 score for each class : success  and failure
                # For success
                precision_success = precision_score(
                    y_true=Y_test, y_pred=y_pred, average="binary", pos_label=1
                )
                recall_success = recall_score(
                    y_true=Y_test, y_pred=y_pred, average="binary", pos_label=1
                )
                f1_success = f1_score(
                    y_true=Y_test, y_pred=y_pred, average="binary", pos_label=1
                )

                # For Failure
                precision_failure = precision_score(
                    y_true=Y_test, y_pred=y_pred, average="binary", pos_label=0
                )
                recall_failure = recall_score(
                    y_true=Y_test, y_pred=y_pred, average="binary", pos_label=0
                )
                f1_failure = f1_score(
                    y_true=Y_test, y_pred=y_pred, average="binary", pos_label=0
                )

                print_predictions(Y_test, y_pred)

                print("")
                print(
                    "Accuracy,ROC_AUC,Precision-Macro,Precision-Micro,Precision-Weighted,Recall-Macro,Recall-Micro,Recall-Weighted,F1-Macro,F1-Micro,F1-Weighted,Failure-Precision,Failure-Recall,Failure-f1,Success-Precision,Success-Recall,Success-F1"
                )
                print(
                    "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
                        test_acc,
                        roc_auc,
                        p_macro,
                        p_micro,
                        p_weighted,
                        r_macro,
                        r_micro,
                        r_weighted,
                        f_macro,
                        f_micro,
                        f_weighted,
                        precision_failure,
                        recall_failure,
                        f1_failure,
                        precision_success,
                        recall_success,
                        f1_success,
                    )
                )

                print(
                    "Feature,{}, weighted F1,{}".format(
                        "::".join(feature) if isinstance(feature, list) else feature,
                        f_weighted,
                    )
                )

