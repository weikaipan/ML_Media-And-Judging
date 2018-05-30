"""prepare news for embedding, and news objects."""
import glob
import pandas as pd

from utility import json_load, text_dump, pickle_dump
from utility import clean_text, normalizeString
from utility import formatDate


class Doc():
    """
    This class is for news mapping:
    news_table, Saves all news title before "date"
                { date: 20170201, news_list: [ title1, title2, ...] }
    """
    def __init__(self):
        self.news_table = {}
        self.title2index = {}
        self.index2title = {}
        self.title2content = {}
        self.n_news = 0

    def add_news(self, title, content, date):
        """
        Args:
            article = {title: {'DATE': date, 'ptime': date,
                       'content': body} }
        """
        k = self.n_news + 1
        self.title2index[title] = k
        self.index2title[k] = title
        self.title2content[title] = {'content': content, 'date': date}

        # date mapping
        if date in self.news_table:
            self.news_table[date]['list'].append(title)
        else:
            self.news_table[date] = {}
            self.news_table[date]['list'] = [title]
        self.n_news += 1
        return


def read_news(old=True, recent=True):
    """Read news data by options."""
    corpus = ""
    doc = Doc()
    if old:
        # Read news csv files
        print("Preparing old news")
        path = '/data/WorkData/media_and_judging/data/collected/nytimes/'
        newsfiles = glob.glob(path + '*.csv')
        print(newsfiles)
        tf = len(newsfiles)
        print("Total old news files: {}".format(tf))

        # concate all cleaned news text
        for news in newsfiles:
            df = pd.read_csv(news)
            count = 0
            for index, row in df.iterrows():
                count += 1
                # append text
                try:
                    cleaned_text = normalizeString(row[5].replace("LEAD: ", ""))
                    corpus += cleaned_text

                    # add to lang row[4]: title, row[5]: content
                    date = formatDate(row[0], '%Y/%m/%d')
                    doc.add_news(row[4], cleaned_text, date)
                except:
                    pass
            if count % 500 == 0:
                print("Progress Report: Old news {}, total {}".
                      format(count, len(news)))
            tf -= 1
            print("Files Remaining: ", tf)

    if recent:
        # Read json
        # {title: {content:, ptime:, utime:, }..}
        print("Preparing recent news")
        path = '/data/WorkData/media_and_judging/data/collected/nytimes/'
        newsfiles = glob.glob(path + '*.json')
        tf = len(newsfiles)
        print("Length of recent news: {}".format(tf))

        # concate all cleaned news text
        for news in newsfiles:
            count = 0
            n = json_load(news)
            for title in n:
                count += 1
                # append text
                cleaned_text = normalizeString(clean_text(n[title]['content']
                                               .replace("LEAD: ", "")))

                corpus += cleaned_text
                # add to lang
                try:
                    if 'ptime' in n[title]:
                        date = formatDate(n[title]['ptime'], '%Y%m%d%H%M%S%f')
                    else:
                        date = formatDate(n[title]['DATE'], '%m. %d, %Y')
                except:
                    date = 'UNKNOWN'
                doc.add_news(title, cleaned_text, date)
            if count % 500 == 0:
                print("Progress Report: Recent news, {}, total {}".
                      format(count, len(news)))
            tf -= 1
            print("Files Remaining: ", tf)

    return corpus, doc


def main():
    """main."""
    print("Read news start")
    corpus, doc = read_news(old=False, recent=True)
    text_dump(corpus, '/data/WorkData/media_and_judging/data/prepared/allnews_8710.txt')
    pickle_dump(doc, '/data/WorkData/media_and_judging/data/prepared/newsobject8710.p')

if __name__ == '__main__':
    main()
