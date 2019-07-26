import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import numpy as np
from pprint import pprint


def collect_results(output_order, result_dir, type="clf"):
    """
     utility to collect results from files
    :param result_dir: The directory that contains all the results files
    :return: None, prints the results in format feature name and then scores all separated by commas
    """

    result_dic = {}
    for result_file in os.listdir(result_dir):
        # print(result_file)
        with open(os.path.join(result_dir, result_file), "r") as f_in:
            lines = f_in.readlines()
            results = lines[-1]
            if type == "reg":
                _, feature_name = lines[0].split(
                    "Running Regression Experiment for Feature:"
                )
            elif type == "clf":
                _, feature_name = lines[0].split("Running Experiment for Feature:")
            else:
                raise NotImplementedError("Type not known")
            feature_name = feature_name.strip()
            output = "{feature_name},{scores}".format(
                feature_name=feature_name, scores=results
            )
            if feature_name.startswith("["):
                # eg  ['concepts_score', 'concepts', 'writing_density_scaled', 'categorical_char_ngram_mid_word']
                feature_name = feature_name.lstrip("[").rstrip("]")
                key = "-".join(
                    [feature.strip().strip("'") for feature in feature_name.split(",")]
                )
                result_dic[key] = output
            else:
                result_dic[feature_name] = output
    # pprint(result_dic)
    print(
        set(result_dic.keys())
        - set(["-".join(f) if isinstance(f, list) else f for f in output_order])
    )
    for f in output_order:
        # print (f)
        if isinstance(f, list):
            key = "-".join(f)
            print(result_dic.get(key, ""))
        else:
            print(result_dic.get(f, ""))


def output_fname_generator(features):
    import hashlib

    if isinstance(features, list):
        fname = "-".join(features)
        hash_object = hashlib.md5(fname.encode("utf-8"))
        hex_dig = hash_object.hexdigest()
        return str(hex_dig)
    else:
        return features


def print_classification_report(Y_actual, Y_pred, classes):
    print("Classifation Report")
    target_names = classes
    class_indices = {kls: i for i, kls in enumerate(target_names)}

    print(
        classification_report(
            Y_actual,
            Y_pred,
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
            Y_actual, Y_pred, labels=[class_indices[cls] for cls in target_names]
        )
    )

    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        Y_actual, Y_pred, average="micro", pos_label=None
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        Y_actual, Y_pred, average="macro", pos_label=None
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        Y_actual, Y_pred, average="weighted", pos_label=None
    )
    roc_auc = roc_auc_score(y_true=Y_actual, y_score=Y_pred)

    # Precision recall f1 score for each class : success  and failure
    # For success
    precision_success = precision_score(
        y_true=Y_actual, y_pred=Y_pred, average="binary", pos_label=1
    )
    recall_success = recall_score(
        y_true=Y_actual, y_pred=Y_pred, average="binary", pos_label=1
    )
    f1_success = f1_score(y_true=Y_actual, y_pred=Y_pred, average="binary", pos_label=1)

    # For Failure
    precision_failure = precision_score(
        y_true=Y_actual, y_pred=Y_pred, average="binary", pos_label=0
    )
    recall_failure = recall_score(
        y_true=Y_actual, y_pred=Y_pred, average="binary", pos_label=0
    )
    f1_failure = f1_score(y_true=Y_actual, y_pred=Y_pred, average="binary", pos_label=0)

    test_acc = accuracy_score(Y_actual, Y_pred) * 100
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


def print_regression_report(Y_actual, Y_pred):
    rmse = np.sqrt(mean_squared_error(Y_actual, Y_pred))
    print(" RMSE : {}".format(rmse))
    r2 = r2_score(Y_actual, Y_pred)
    print(" R2 : {}".format(r2))

