# -*- coding: utf-8 -*-
from __future__ import print_function, division

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from contextlib import redirect_stdout
from author2vec.experiments import get_classifiers
from author2vec.manage import app

from author2vec.features.utils import fetch_features_vectorized
from author2vec.experiments.utils import output_fname_generator
import os
import tqdm


def genre_classification(corpus, feature, dump_dir=app.VECTORS, ignore_lst=None):

    # classifiers=['lr','rf','percepton','svc']
    classifiers = ["svc"]

    for classifier in tqdm.tqdm(classifiers):
        print(classifier)

        clf_algorithm, param_grid = get_classifiers(classifier)

        out_file = output_fname_generator(feature)
        out_file = "{}_{}.txt".format(out_file, classifier)
        with open(os.path.join(app.GENRE_OUTPUT, out_file), "w") as f:

            with redirect_stdout(f):
                X_train, Y_train, X_test, Y_test = fetch_features_vectorized(
                    dump_dir, feature, corpus
                )
                classes = corpus.labels()

                # X_train ,Y_train ,X_test ,y_test ,classes   =  split_train_test(base_path=dir_path ,genre=genre)
                print("Total dataset size:")
                print("n_samples traning: %d" % len(X_train))

                print("n_classes: %d" % len(set(Y_train)))

                print("test size %d" % len(Y_test))
                # Train a SVM classification model

                print("Fitting the classifier to the training set")

                # param_grid = {'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10, 100, 1000, 10000]}

                # clf_algorithm= LogisticRegression() #LinearSVC()

                clf = GridSearchCV(clf_algorithm, param_grid, scoring="f1_weighted")
                clf = clf.fit(X_train, Y_train)

                print("Best estimator found by grid search:")
                print(clf.best_estimator_)

                print("Best score: %0.3f" % clf.best_score_)
                print("Best parameters set:")
                best_parameters = clf.best_estimator_.get_params()
                for param_name in sorted(param_grid.keys()):
                    print("\t%s: %r" % (param_name, best_parameters[param_name]))

                for params, mean_score, scores in clf.grid_scores_:
                    print("%0.3f+/-%0.2f %r" % (mean_score, scores.std() / 2, params))

                ###############################################################################
                # Quantitative evaluation of the model quality on the test set

                print("Predicting  on the test set")

                y_pred = clf.predict(X_test)

                # print (Y_test)
                #
                # print (y_pred)

                target_names = classes.tolist()
                print(target_names)
                class_indices = {cls: idx for idx, cls in enumerate(target_names)}

                print(classification_report(Y_test, y_pred))
                print(confusion_matrix(Y_test, y_pred))

                #
                accuracy = accuracy_score(Y_test, y_pred) * 100
                print("============================================================")
                print("Accruacy (%) :", accuracy)
                print("============================================================")

                p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
                    Y_test, y_pred, average="micro", pos_label=None
                )
                p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
                    Y_test, y_pred, average="macro", pos_label=None
                )
                p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
                    Y_test, y_pred, average="weighted", pos_label=None
                )
                # roc_auc_micro, roc_auc_macro, roc_auc_weighted = roc_auc_score(y_true=Y_test, y_score=y_pred,average='micro'),roc_auc_score(y_true=Y_test, y_score=y_pred,average='macro'),roc_auc_score(y_true=Y_test, y_score=y_pred,average='weighted')

                print()
                print("============================================================")
                print(
                    "Macro Precision score  %f , Micro Score %f , Weighted Score %f :"
                    % (
                        precision_score(
                            Y_test, y_pred, average="macro", pos_label=None
                        ),
                        precision_score(
                            Y_test, y_pred, average="micro", pos_label=None
                        ),
                        precision_score(
                            Y_test, y_pred, average="weighted", pos_label=None
                        ),
                    )
                )
                print("============================================================")
                print(
                    "Macro Recall score  %f , Micro Score %f , Weighted Score %f:"
                    % (
                        recall_score(Y_test, y_pred, average="macro", pos_label=None),
                        recall_score(Y_test, y_pred, average="micro", pos_label=None),
                        recall_score(
                            Y_test, y_pred, average="weighted", pos_label=None
                        ),
                    )
                )
                print("============================================================")
                print(
                    "Macro F1score  %f , Micro F1-Score %f , Weighted Score %f:"
                    % (
                        f1_score(Y_test, y_pred, average="macro", pos_label=None),
                        f1_score(Y_test, y_pred, average="micro", pos_label=None),
                        f1_score(Y_test, y_pred, average="weighted", pos_label=None),
                    )
                )
                print("============================================================")

                print(
                    "Accuracy,Precision-Macro,Precision-Micro,Precision-Weighted,Recall-Macro,Recall-Micro,Recall-Weighted,F1-Macro,F1-Micro,F1-Weighted"
                )
                print(
                    "{},{},{},{},{},{},{},{},{},{}".format(
                        accuracy,
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

