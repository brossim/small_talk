# Automated Detection of Small Talk in Public Service Discourse

## Final Project for the Seminar "Einführung in Textklassifikation" by Simon Bross (809648)

This project focuses on developing a Support Vector Machine (SVM) classifier to automatically detect small talk in spoken conversation from Public Service Discourse. 

## 1. Project Structure & Components

```
Project Directory
|── data
|── error_analysis
|── reports
|-- experiments.py
|-- features.py
|-- main.py
|-- requirements.txt
|-- svm_finetuning.py

```

### 1.1 'data' Directory 
Contains a script (get_data.py) to retrieve the relevant data from study_ann_3.csv. Furthermore, it provides a comprehensive stop word list for German. 

### 1.2 'error_analysis' Directory

Contains the confusion matrices stored for every model configuration and evaluation fold from k-fold cross validation. 

### 1.3 'reports' Directory

Contains the averaged results from 5-fold cross validation on every model configuration. 

### 1.4 experiments.py 

Implements the functionality to run a k-fold cross validation experiment. 

### 1.5 features.py 

Implements the FeatureExtractor class used for feature engineering. 

### 1.6 main.py

Main script from which the experiments are run. 

### 1.7 requirements.txt
Lists all necessary dependencies for the project. Dependencies can be installed from the terminal using the following command: 
```
$ pip install -r requirements.txt 
```

### 1.8 svm_finetuning.py

Implements the functionality to perform hyperparameter tuning and subsequent evaluation of the SVM model. 

### 2. Python 

Python 3.9 was used for this project. 

### 3. Data

The first annotation round for small talk in the PSE corpus serves as the data for this project, encompassing 2002 labeled utterances.  