# Featurization

## Description

## Requirements
1. nltk 3.2.5

## Scripts

```preparenews.py```
	
    Input: all raw news in /data/WorkData/media_and_judging/collected/nytimes.
    
    Output:
    newsobject8710.p


Aggregates raw news data with date from 1987 to 2010, and generates all news metadata and information. It then stores news data into a self-defined object for latter look up.

```addnewswithintimeframe.py```

    Input: court_judge_case_trend_wthr_Data_inner_reDefTrend.csv, newsobject8710.p

    Output: court_news_list_90days_0110.csv'

This script takes data from Jess Eagel's paper with adjudication date columns. It then adds a new column containing a list of news articles 90 days before the asylum cases.

```documentEmbedding.py```

	Input:
	newsobject8710.p, court_news_list_90days_0110.csv, news_title2embedding8710.p

	Output:
    final_grouped500_news_court_trend_0110_test.csv

This script accepts the object storing news metadata from 1987 to 2010, and generates vector representations of article using word vectors in ```news8710.vec``` which is trained using fasttext by skip gram on concatenation of all news articles from 1987 to 2010. The script then applies k means clustering to cluster all article vectors into 500 groups by default.

```cable_toEmbedding_toGroup.ipynb```

This ipython notebook applies the same method as ```documentEmbedding.py``` to Wikileaks cables, and group them into 100 groups of cables by content similarity after representing articles as vectors.


```generate_new_trend_features.py```

    Input:
    /data/WorkData/media_and_judging/data/prepared/court_judge_case_trend_wthr_Data_inner_sorted.csv

    Output:
    /data/WorkData/media_and_judging/data/prepared/court_judge_case_trend_wthr_Data_inner_reDefTrend.csv

This script takes Jess Eagel's dataset sorted by adjudication time, and generates two new trend features: 

* ```judge_grantRatio_prevMonth``` = 

    \#of grant / # of total cases by judge in previous month of adj_date 
 
* ```court_grantRatio_prevMonth``` = 

    \#of grant / # of total cases by court in previous month of adj_date






