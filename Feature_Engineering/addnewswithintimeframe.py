"""."""

import pandas as pd
import datetime

from datetime import timedelta
from utility import pickle_load
from preparenews import Doc


def gen_date_range(court_case_date, time_frame=7):
    """A helper function to genereate a list of a range of dates"""
    end_date = datetime.datetime.strptime(court_case_date, "%Y%m%d")
    start_date = end_date - timedelta(days=time_frame) + timedelta(days=1)
    return [str(m) for m in pd.date_range(start_date, end_date, freq='1D')]


def addNewsinTimeFrame(NewsLang, df, time_frame=30):
    """
    Args:
        df = court case dataframe
        NewsLang = News object, containing title2news, date2news_list, idx2news, title2content 
    Returns:
        df:
            dataframe joined with a column news_date_index
        news_date_index:
            a list of news lists mapping to index of news title
            [[1:2:3:4:], [:::]...]
        news_date_map:
            Same as news_date_index, while the index is mapped to title string
    """
    news_date_index = []
    # news_ngram_df = pd.DataFrame()
    print("Begin Processing news within timeframe {d} days before the date each court case".format(d=time_frame))
    case_number = 0
    for court_case_date in df['Case_Adj_Date']:
        case_number = case_number + 1
        daterange = gen_date_range(court_case_date, time_frame)
        # append news title within the time frame to the court case
        index_str = ""
        for d in daterange:
            d2Date = datetime.datetime.strptime(d.split()[0], "%Y-%m-%d").strftime("%Y%m%d")
            if d2Date in NewsLang.news_table:
                """
                check the date is in the news table beforehand
                news_table[date_to_map] = {
                    news_list: [title1, title2, ...]
                }
                """
                for title in NewsLang.news_table[d2Date]['list']:
                    index_str += str(NewsLang.title2index[title]) + ':'

        news_date_index.append(index_str)
        # progress reporting
        if case_number % 1000 == 0:
            print("Working on {}".format(case_number))

    df['nytimes_' + str(time_frame) + 'day_index'] = news_date_index
    print("Complete Merging Court and Time frame")
    print("Dataframe shape = ", df.shape)
    print(df.columns.values)
    return df


def add_date_string(df):
    """
    This function concat case completion date and adjudication date
    into a single column, and keep comp_year, adj_year.
    """
    adj_date_col = []
    for y, m, d in zip(df['adj_year'], df['adj_month'], df['adj_day']):
        adj_date_col.append(datetime.datetime(year=y, month=m, day=d).strftime("%Y%m%d"))
    df['Case_Adj_Date'] = adj_date_col
    return df


def sample_dataframe(df, begin_year, end_year):
    """
    Args:
    pandas dataframe
    Return:
    Either a full size dataframe or sampled dataframe by time or size
    """
    df = df[(df['adj_year'] >= begin_year) & (df['adj_year'] <= end_year)]
    print("Sampled year = {y1} to {y2}, Size of = {l}, Datatype = pandas.Dataframe".format(y1=begin_year, y2=end_year, l=df.shape))
    return df


def read_main(begin_year, end_year):
    print("Load Court Case as Dataframe...")
    df = pd.read_csv('/data/WorkData/media_and_judging/data/prepared/court_judge_case_trend_wthr_Data_inner_reDefTrend.csv', encoding="ISO-8859-1",
                     error_bad_lines=False, warn_bad_lines=True)
    print("Court Case Shape: ", df.shape)
    df = add_date_string(df)
    df = sample_dataframe(df, begin_year, end_year)
    print(df.columns.values)
    return df


def main():
    # The main court case data
    df = read_main(2001, 2010)
    # All news from 1987 to 2010
    doc = Doc()
    doc = pickle_load('/data/WorkData/media_and_judging/data/prepared/newsobject8710.p')
    # Add time range column for each court case
    df = addNewsinTimeFrame(doc, df, time_frame=90)
    df.to_csv('/data/WorkData/media_and_judging/data/prepared/court_news_list_90days_0110.csv')

if __name__ == '__main__':
    main()
