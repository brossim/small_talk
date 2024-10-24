# Author        : Simon Bross
# Date          : June 15, 2024
# Python version: 3.9
# Performs hyperparameter tuning on SVM classifier

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from data.get_data import get_x_y
from features import FeatureExtractor
from sklearn.model_selection import StratifiedKFold
from experiments import run_experiment

extractor = FeatureExtractor()


def finetune_svm(features, with_context=False):
    """
    Fine-tunes an SVM model on the data using contextual utterances and
    a specified feature (combination).
    (c.f. features.py).
    This function performs the following steps:
    1. Loads the data with/without context.
    2. Extracts features.
    3. Defines a parameter grid for hyperparameter tuning.
    4. Uses GridSearchCV to find the best hyperparameters for the SVM model.
    5. Runs k-fold cross-validation and evaluation using the best config.
    @param features: A list of features to be extracted, or 'all' to use all
    features implemented in features.py.
    @param with_context: Boolean determining whether to load the data with
    contextual utterances or not.
    """
    X, y = get_x_y(with_context=with_context)
    X_train_plain, X_test_plain, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )
    X_train, X_test = extractor.get_features(
        X_train_plain, X_test_plain,
        select=features
    )
    param_grid = {
        'C': [0.1, 0.5, 1, 2, 3],
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'gamma': ["scale", "auto", 1, 0.1, 0.01, 0.001, 0.0001],
        'class_weight': ['balanced', None]
    }
    svc = SVC()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    grid = GridSearchCV(estimator=svc, param_grid=param_grid, cv=kf,
                        verbose=2, n_jobs=1, scoring="f1_macro")
    grid.fit(X_train, y_train)
    print("Best parameters found:", grid.best_params_)
    # Perform k-fold cross-validation with the best configuration
    model = grid.best_estimator_
    run_experiment(extractor, features=features,
                   with_context=with_context, tuned_model=model)
    print("Running k-fold cross validation and evaluation using the tuned "
          "hyperparameters")


if __name__ == "__main__":
    # finetune best and worst performing configuration (acc. to macro avg. F1)
    # from experiments
    # 1. best: EMB, POS, COM, SENT, with context
    finetune_svm(features=["embedding", "pos", "complexity", "sentiment"],
                 with_context=True)
    # 2. worst: TFIDF, POS, COM, SENT, without context
    finetune_svm(features=["tf_idf", "pos", "complexity", "sentiment"],
                 with_context=False)
