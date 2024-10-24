# Author        : Simon Bross
# Date          : June 3, 2024
# Python version: 3.9

# spacy model download: python -m spacy download de_dep_news_trf

import numpy as np
import re
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from textstat import textstat
from tqdm import tqdm
from collections.abc import Iterable
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from data.stop_words import german_stopwords


class FeatureExtractor:
    """
    A class for extracting various linguistic features from textual data.
    The features comprise tf-idf representation, POS tags, sentence embeddings,
    sentiment, and text complexity (Wiener Sachtextformel).
    """
    def __init__(self):
        """
        Initializes the FeatureExtractor with the necessary models and spacy
        pipeline.
        """
        # load spacy and embedding model during initialization so that
        # they do not have to be reloaded for every fold in k-fold
        print("Loading Spacy Model. If not yet downloaded, "
              "use: $ python -m spacy download de_core_news_md")
        self.__nlp = spacy.load("de_core_news_md")
        self.__nlp.add_pipe("spacytextblob")
        print("Loading SentenceTransformer model. "
              "Will be downloaded, if necessary.")
        self.__embedding_model = SentenceTransformer(
            'paraphrase-multilingual-MiniLM-L12-v2'
        )

    @staticmethod
    def __preprocess_utterances(utterances):
        """
        Preprocess a list of utterances by removing specific annotations and
        symbols. This method performs the following preprocessing steps on
        each utterance:
        1. Removes annotations enclosed in round brackets (e.g., pauses and
        other annotations).
        2. Removes square brackets '[' and ']', and both single '/' and
        double '//' slashes.
        @param utterances: Utterances to be preprocessed as a list of strings.
        @return: Preprocessed utterances as a list of strings.
        """
        # remove annotation of pauses and other annotation using round brackets
        round_brackets = re.compile(r"\([ .a-zA-Z0-9äöüÄÖÜß-]+\)")
        # remove '[' and ']' and both '/' and '//'
        square_brackets = re.compile(r"[\[|\]]")
        slashes = re.compile(r"[/|//]")
        X_new = []
        for utterance in utterances:
            utterance = round_brackets.sub("", utterance)
            utterance = square_brackets.sub("", utterance)
            utterance = slashes.sub("", utterance)
            X_new.append(utterance)
        return X_new

    @staticmethod
    def __check_data_constitution(training_data, test_data):
        """
        Ensures that the training and test data are correctly formatted
        as lists of strings.
        @param training_data: Training dataset as a list of strings.
        @param test_data: Test dataset as a list of strings.
        """
        assert isinstance(training_data, Iterable)
        for sent in training_data:
            assert isinstance(sent, str)
        assert isinstance(test_data, Iterable)
        for sent in test_data:
            assert isinstance(sent, str)

    def get_features(self, training_set, test_set, select="all"):
        """
        Extracts various features from the data for the training and test set.
        This method allows the extraction of different features based on
        the specified features. Features include:
            - 'pos': Count vectorized matrix for part-of-speech tags.
            - 'sentiment': Sentiment scores of the utterances.
            - 'embedding': Sentence embeddings of the utterances.
            - 'tf_idf': TF-IDF matrix.
            - 'complexity': Text complexity scores (Wiener Sachtextformel).
        @param training_set: List of strings representing the training dataset.
        @param test_set: List of strings representing the test dataset.
        @param select: Defaults to "all", thus extracting all features.
               Alternatively, a list of strings can be provided to select a
               subset of the available features.
        @return: A tuple containing two NumPy arrays:
           - feature_matrix_train (np.ndarray): Feature matrix (training data).
           - feature_matrix_test (np.ndarray): Feature matrix (test data).
        """
        self.__check_data_constitution(training_set, test_set)
        print("Preprocessing textual data")
        train_pre = self.__preprocess_utterances(training_set)
        test_pre = self.__preprocess_utterances(test_set)
        # nlp pipe only necessary if features != embedding and/or complexity
        if select not in (
                ["embedding"], ["complexity"], ["embedding", "complexity"],
                ["complexity", "embedding"]
        ):
            print("Running spacy nlp pipe")
            # join data to avoid using two nlp pipes
            # no data leakage for POS-tagging/sentiment/lemmatization
            # if features are embedding and/or complexity, do not run pipe
            docs = self.__nlp.pipe(train_pre + test_pre)
            # get train_data length to re-split data into train and test
            train_idx = len(training_set)
            docs_list = list(docs)
            train_docs = docs_list[:train_idx]
            test_docs = docs_list[train_idx:]
        # list of all features to
        features = [
            "pos", "sentiment", "embedding", "tf_idf", "complexity"
        ]
        # collect feature arrays for train and test data
        feature_arrays_train = []
        feature_arrays_test = []
        if select != "all":
            assert isinstance(select, list), \
                "'Select' parameter must be set to 'all' or provide a list" \
                " of strings corresponding to the desired features "
            # check if features in 'select' are valid
            assert all(
                [True if feature in features else False for feature in select]
            ), "Invalid feature(s) found in 'select'"
            # sort out features that do not need to be computed
            features = [feature for feature in select]
        for feature in tqdm(
                features,
                desc=f"Extracting {len(features)} feature(s) from the data"
        ):
            # embedding and complexity do not use spacy
            if feature in ["embedding", "complexity"]:
                feature_train, feature_test = \
                    getattr(self, '_' + feature)(train_pre, test_pre)
            else:
                feature_train, feature_test = \
                    getattr(self, '_' + feature)(train_docs, test_docs)
            feature_arrays_train.append(feature_train)
            feature_arrays_test.append(feature_test)

        # concatenate all features to a single feature matrix
        feature_matrix_train = np.concatenate(feature_arrays_train, axis=1)
        feature_matrix_test = np.concatenate(feature_arrays_test, axis=1)
        # feature scaling using StandardScaler
        scaler = StandardScaler()
        scaler.fit(feature_matrix_train)
        X_train_scaled = scaler.transform(feature_matrix_train)
        X_test_scaled = scaler.transform(feature_matrix_test)
        return X_train_scaled, X_test_scaled

    def _pos(self, training_docs, test_docs):
        """
        Applies POS-tagging on the training and test data. The frequency
        of every POS tag in a sentence is counted using count-vectorization.
        @param training_docs: List of SpaCy Doc objects for training data.
        @param test_docs: List of SpaCy Doc objects for test data.
        @return: A tuple containing two elements:
           - matrix_train (numpy.ndarray): POS matrix (training data).
           - matrix_test (numpy.ndarray): POS matrix (test data).
        """
        train_tagged = [
            [word.pos_ for word in doc] for doc in training_docs
        ]
        test_tagged = [
            [word.pos_ for word in doc] for doc in test_docs
        ]
        # vectorize the POS tags using CountVectorizer
        # tokenizer and preprocessor are not needed, data already tokenized
        cv = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
        matrix_train = cv.fit_transform(train_tagged).toarray()
        matrix_test = cv.transform(test_tagged).toarray()
        return matrix_train, matrix_test

    def _sentiment(self, training_docs, test_docs):
        """
        Analyzes sentiment using the spacytextblob extension. Computes the
        sentiment score ranging from [-1, 1] for each sentence in the
        training and test data.
        @param training_docs: List of SpaCy Doc objects for training data.
        @param test_docs: List of SpaCy Doc objects for test data.
        @return: A tuple containing two elements:
            - sentiment_train (numpy.ndarray): Sentiment matrix (training data)
            - sentiment_test (numpy.ndarray): Sentiment matrix (test data).
        """
        sentiment_train = np.array(
            [[doc._.blob.polarity] for doc in training_docs]
        )
        sentiment_test = np.array(
            [[doc._.blob.polarity] for doc in test_docs]
        )
        return sentiment_train, sentiment_test

    def _embedding(self, training_sents, test_sents):
        """
        Generates sentence embeddings using the
        'distiluse-base-multilingual-cased-v2' model from the
        sentence transformers library to encode the training
        and test data into dense (512 dimensions) vector representations.
        @param training_sents: List of strings representing training sentences.
        @param test_sents: List of strings representing test sentences.
        @return: A tuple containing two elements:
           - train_embedds (np.ndarray): Embedding matrix (training data)
           - test_embedds (np.ndarray): Embedding matrix (test data).
        """
        train_embedds = self.__embedding_model.encode(
            training_sents, convert_to_numpy=True)
        test_embedds = self.__embedding_model.encode(
            test_sents, convert_to_numpy=True)
        return train_embedds, test_embedds

    def _tf_idf(self, training_docs, test_docs):
        """
        Compute TF-IDF vectors for the training and test data. Applies
        sublinear TF scaling which replaces term frequency with
        1 + log(tf), thus assigning less importance to terms that occur
        very frequently. Setting max_df helps in removing terms that are
        too frequent and may not be relevant.
        @param training_docs: List of SpaCy Doc objects for training data.
        @param test_docs: List of SpaCy Doc objects for test data.
        @return: A tuple containing two elements:
            - tf_idf_train (numpy.ndarray): Sparse matrix of TF-IDF vectors
              for the training data.
            - tf_idf_test (numpy.ndarray): Sparse matrix of TF-IDF vectors
              for the test data.
        """
        # lemmatize sentences
        lemma_train = [
            [word.lemma_ for word in doc] for doc in training_docs
        ]
        lemma_test = [
            [word.lemma_ for word in doc] for doc in test_docs
        ]
        # rejoin lemmata as sentences
        train_rejoined = [
            " ".join([tok for tok in sent]) for sent in lemma_train
        ]
        test_rejoined = [
            " ".join([tok for tok in sent]) for sent in lemma_test
        ]
        vectorizer = TfidfVectorizer(
            sublinear_tf=True, max_df=0.5, stop_words=german_stopwords)
        tf_idf_train = vectorizer.fit_transform(train_rejoined).toarray()
        tf_idf_test = vectorizer.transform(test_rejoined).toarray()
        return tf_idf_train, tf_idf_test

    def _complexity(self, training_sents, test_sents):
        """
        Computes text complexity scores for the training and test data using
        the Wiener Sachtextformel from the textstat library. It assesses the
        readability and complexity of German texts by considering factors such
        as sentence length, word length, and syllable count.
        @param training_sents: List of training sentences (str).
        @param test_sents: List of test sentences (str).
        @return: A tuple containing two elements:
           - complexity_train (numpy.ndarray): Array of text complexity scores
             for the training data.
           - complexity_test (numpy.ndarray): Array of text complexity scores
             for the test data.
        """
        textstat.set_lang("DE_ger")
        complexity_train = [
            [textstat.wiener_sachtextformel(text, variant=1)]
            for text in training_sents
        ]
        complexity_test = [
            [textstat.wiener_sachtextformel(text, variant=1)]
            for text in test_sents
        ]
        return np.array(complexity_train), np.array(complexity_test)
