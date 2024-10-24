# Author        : Simon Bross
# Date          : June 2, 2024
# Python version: 3.9
# Main script where experiments are run from

from experiments import run_experiment
from features import FeatureExtractor

# define feature combinations for experiments
feat_comb1 = ["embedding"]
feat_comb2 = ["tf_idf"]
feat_comb3 = ["embedding", "pos", "complexity", "sentiment"]
feat_comb4 = ["tf_idf", "pos", "complexity", "sentiment"]
all_combs = [feat_comb1, feat_comb2, feat_comb3, feat_comb4]


# initialize feature extractor here so that it does not have to
# be re-initialized for every function call
# (loads spacy and embedding model during initialization)
feature_extractor = FeatureExtractor()

# run experiment for every feature combination and
# with/without contextual utterances, fine-tuned or not
if __name__ == '__main__':
    for comb in all_combs:
        run_experiment(
            feature_extractor, features=comb,
            error_analysis=True, with_context=True)
        run_experiment(
            feature_extractor, features=comb,
            error_analysis=True, with_context=False)
