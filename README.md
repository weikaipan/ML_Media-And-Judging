# Media-And-Judging

## Description
This is an academic project of Spring 2018, DS-GA 1003 Machine Learning and Computational Statistics at New York University.

## Requirments
1. Python 3.6
2. nltk 3.2.5
3. Scikit-learn 0.19.1

## Data set 

The final pickle dataset for training models is located at: 


    /data/WorkData/media_and_judging/data/train/final_grouped500_news_court_trend_0110.p   # Random Forest

    /data/WorkData/media_and_judging/data/train/cable_from_to_president_df_2001-2010_group100_90d.p  # Decision Tree

## Usage

1. Run training model: (Using final dataset)
	
	```$ sh runtrain.sh```
	
	The model result will be output to  ``train/model_result/[DecisionTree | RandomForest]/``. 
	
	The feature importance, and ROC_AUC accuracy for each model setting is saved under this folder for each running.


## Result

We used random forest to traing on data with feature sets: asylum court case and news group in time series manner. The results can be found in the file on Azure server:

```/data/WorkData/media_and_judging/model_results/RandomForest/feature_info_final_new_trend_news_group_40_score.csv```

with 0.89 roc auc score after rounding.


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
