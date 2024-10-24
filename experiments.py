# Author        : Simon Bross
# Date          : June 6, 2024
# Python version: 3.9

import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, \
    ConfusionMatrixDisplay
from data.get_data import get_x_y
from sklearn.svm import SVC
from collections import defaultdict


def run_experiment(
        extractor, features="all", error_analysis=True,
        with_context=False, tuned_model=None):
    """
    Performs a classification experiment using SVM and
    stratified k-fold cross-validation, given a set of features that represent
    the data. For evaluation, a classification report is generated after every
    fold and averaged over all folds (and stored in the reports directory).
    If desired, a confusion matrix will be generated and stored for every fold
    (in the error_analysis folder, if error_analysis is set to True).
    @param extractor: FeatureExtractor instance.
    @param features: Features to extract from the data. Defaults to 'all'.
    Alternatively, a subset of the available features can be passed via
    a list of strings (cf. features.py).
    @param error_analysis: Whether to store the confusion matrices (as pdf).
    @param with_context: Boolean indicating whether to include the contextual
    utterances into the data.
    @param tuned_model: Passes a hyperparameter tuned SVM model to be evaluated.
    """
    # Retrieve data
    X, y = get_x_y(with_context=with_context)
    tuned = True if tuned_model is not None else False
    # store metrics for every fold to average over them afterwards
    metrics = defaultdict(lambda: defaultdict(float))
    # use stratified K fold to ensure a well-balanced class distribution
    skf = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(X, y), start=1):
        print(f"\nFold number: {i}\n")
        X_train_plain = [X[p] for p in train_index]
        y_train = [y[p] for p in train_index]
        X_test_plain = [X[p] for p in test_index]
        y_test = [y[p] for p in test_index]
        # extract features for training and test data
        X_train, X_test = extractor.get_features(
            X_train_plain, X_test_plain, select=features
        )
        # use fine-tuned model if passed (via svm_finetuning.py)
        if tuned_model is not None:
            model = tuned_model
        else:
            model = SVC()
        print("Fitting SVM model")
        model.fit(X_train, y_train)
        print("Testing and evaluating SVM model")
        y_pred = model.predict(X_test)

        # get classifcation report and store the values in the metrics dict
        report = classification_report(y_test, y_pred, output_dict=True)
        for class_label, class_metrics in report.items():
            if class_label == 'accuracy':
                metrics['accuracy']['accuracy'] += class_metrics
            elif class_label in ['macro avg', 'weighted avg']:
                for metric, value in class_metrics.items():
                    metrics[class_label][metric] += value
            else:
                for metric, value in class_metrics.items():
                    metrics[class_label][metric] += value
        # get confusion matrices and store them
        if error_analysis:
            cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=model.classes_
            )
            store_plot(disp, i, features, with_context, tuned)

    # average the metrics over the number of folds
    # then store averaged report
    num_folds = skf.n_splits
    results = {}
    for class_label, class_metrics in metrics.items():
        results[class_label] = {
            metric: value / num_folds for
            metric, value in class_metrics.items()
        }
    store_results(str(results), features, with_context, tuned)


def store_plot(plot, fold_num, features, context, tuned):
    """
    Saves the confusion matrix as a .pdf file in the 'error_analysis'
    directory.
    The plot is named according to the fold number, the features used,
    whether contextual utterances were included, and whether a finetuned
    model is used.
    @param plot: ConfusionMatrixDisplay instance.
    @param fold_num: Fold number of k-fold cross validation as int,
    indicating which iteration of the process produced the plot.
    @param features: The features that were extracted from the data,
    either all available (='all') or a subset as a list of strings.
    @param context: Boolean indicating whether context was included in
    the features.
    @param tuned: Boolean determining whether a fine-tuned model was used.
    """
    tuned_str = "tuned" if tuned else "non_tuned"
    context = "with_context" if context else "without_context"
    # check if directory exists, create if not
    if not os.path.isdir("error_analysis"):
        os.mkdir("error_analysis")
    # determine file name based on parameters
    if features != "all":
        file_name = f"fold_{fold_num}_{'_'.join(features)}_" \
                    f"{context}_{tuned_str}.pdf"
    else:
        file_name = f"fold_{fold_num}_{context}_" \
                    f"all_{tuned_str}.pdf"
    file_path = os.path.join("error_analysis", file_name)
    plot.plot(values_format="d")
    plt.savefig(file_path)
    plt.close()


def store_results(report, features, context, tuned):
    """
    Saves the averaged classification report (over k-folds) as a .txt file
    in the 'reports' directory. The report is named according
    the features used, whether contextual utterances were included, and
    whether a finetuned model was used.
    @param report: Averaged report results as str.
    @param features: The features that were extracted from the data, either all
    available (='all') or a subset as a list of strings.
    @param context: Boolean indicating whether contextual utterances were
    included.
    @param tuned: Boolean determining whether a fine-tuned model was used.
    """
    tuned_str = "tuned" if tuned else "non_tuned"
    context = "with_context" if context else "without_context"
    # check if directory exists, create if not
    if not os.path.isdir("reports"):
        os.mkdir("reports")
    # determine file name based on parameters
    if features != "all":
        file_name = f"{'_'.join(features)}_{context}_{tuned_str}.txt"
    else:
        file_name = f"{context}_all_features_{tuned_str}.txt"
    file_path = os.path.join("reports", file_name)
    with open(file_path, "w") as out:
        out.write(report)
