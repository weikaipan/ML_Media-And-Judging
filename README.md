# Media-And-Judging

## Description
This is a binary classification task using machine learning models to predict judge's adjudication on asylum court. We further generated and quantified textual features extracted from New York Times and Wikileaks corpus using NLP tools including fasttext for language model, n-grams, and tf-idf.

In ```feature_engineering```  folder, we presented a way to reduce the dimension and sparsity of textual features using language model and k-means clustering. In ```train``` folder, we are developing a machine learning task abstraction which aims to be reused for following tasks. The script for task abstraction is in ```./train/model.py```, and the driver script is ```./train/train.py```.

## Requirments
1. Python 3.6
2. fasttext
3. Scikit-learn 0.19.1


## Usage

1. Run training model: (Using final dataset)
	
	```$ sh runtrain.sh```
	
	The model result will be output to  ``train/model_result/[DecisionTree | RandomForest]/``. 
	
	The feature importance, and ROC_AUC accuracy for each model setting is saved under this folder for each running.


## Result

We used random forest to traing on data with feature sets: asylum court case and news group in time series manner.

We reached 0.89 roc auc score after rounding.


## Table of Content

* [**Data-Preprocessing**](https://github.com/Machine-Learning-NYU-2018/Media-And-Judging/blob/master/Data-Preprocessing/README.md): 
	1. Scrape nytimes
	2. Scrape wikileaks
	3. Join case data

* [**Featurization**:](https://github.com/Machine-Learning-NYU-2018/Media-And-Judging/tree/master/Featurization/README.md)
    1. Feature Engineering

* [**train**:](https://github.com/weikaipan/Media-And-Judging/blob/master/train/README.md) 
	1. Training pipeline starts here, the scripts only support DecisionTree and RandomForest now
	2. Please make sure directories are created : ```model_results/[DecisionTree | RandomForest]``` under the train directory

* [**analysis**:](https://github.com/weikaipan/Media-And-Judging/blob/master/analysis/README.md) 
	1. Model analysis, and findings in data
