"""Main training script."""
import pandas as pd

from all_feature_list import news_features_500, cables_dir, cable_unigrams, object_columns
from all_feature_list import unigram, bigrams, court_cases_all, new_trend, own_selection1
from all_feature_list import case, judge, court, cable_groups, object_columns_bad, cable_to_drop

from utility import pickle_dump, pickle_load
from model import MLTask

# import graphviz
def modeling():
    """."""
    news_group500_path = '/data/WorkData/media_and_judging/data/train/final_grouped500_news_court_trend_0110.p'
    cable_100_path = '/data/WorkData/media_and_judging/data/train/cable_from_to_president_df_2001-2010_group100_90d.p'

    # Random Forest, News Group Training.
    print("News Group Training Random Forest.")
    df = pd.read_pickle(news_group500_path)
    y = df['grant']
    cv_modeling(df.drop(['grant'] + object_columns, axis=1), y,
                'RandomForest', estimators_attemp=[10, 20, 30, 40, 50],
                depth_attemp=[4, 6, 8, 10], split_attemp=[2, 4, 8, 16])

    # Decision Tree, Cable Group Training.
    print("Cable Group Decision Tree.")
    df = df.drop(news_features_500, axis=1)
    cable = pd.read_pickle(cable_100_path)
    cable = pd.merge(df, cable, how='left', on=['idncase', 'idnproceeding'])
    print("Merged DaraFrame: {}".format(cable.shape))

    cable.fillna(0, inplace=True)
    y = cable['grant']
    cable.drop(['grant'] + cable_to_drop, axis=1, inplace=True)
    print("Data Shape = ", cable.shape)
    cv_modeling(cable, y, 'DecisionTree', depth_attemp=[4, 6, 8, 10])
    return


def cv_modeling(df, y, model, estimators_attemp=[10], depth_attemp=[5], split_attemp=[2]):
    """."""
    if model == 'DecisionTree':
        for depth in depth_attemp:
            cable_trend = MLTask(df[cable_groups + cables_dir + new_trend + court_cases_all], y,
                                 model_type='clf', model_name=model, plot=False,
                                 cross_validate=False, standardized=False,
                                 normalized=False, task_name='Cable_trend_court', params=None,
                                 time_series=True, verbose=False, depth=depth)
            cable_trend.batch()
    else:
        for estimators in estimators_attemp:
            for depth in depth_attemp:
                for split in split_attemp:
                    print("Estimators = {}".format(estimators))
                    news_trend = MLTask(df, y,
                                        model_type='clf', model_name=model, plot=False,
                                        cross_validate=False, standardized=False,
                                        normalized=False, task_name='final_new_trend_news_group_', params=None,
                                        time_series=True, verbose=False, self_split=True,
                                        estimators=estimators, depth=depth, split=split)
                    news_trend.batch()


def main():
    """Main driver for training."""
    print("Experiments")
    modeling()
    print("Experiments Done.")

if __name__ == '__main__':
    main()
