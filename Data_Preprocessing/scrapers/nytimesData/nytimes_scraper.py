"""
Reference:

"""
import requests
import json
import time
import sys
import unicodedata
import string
import re
import numpy as np
import datetime
from datetime import timedelta
from apikeys import key
from bs4 import BeautifulSoup  
import NYTapi

"""
Referenced from:
http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

import unicodedata
import string
import re

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def remove_non_ascii(text):
    """
    Note:
     ord('a') returns the integer 97 , ord(u'\u2020') returns 8224 .
    """
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = remove_non_ascii(s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?']+", r" ", s)
    return str(s.encode('utf-8').decode('ascii', 'ignore'))


def parse_articles(articles):
    '''
    This function takes in a response to the NYT api and parses
    the articles into a list of dictionaries
    '''
    news = []
    try:
        for i in articles['response']['docs']:
            dic = {}
            dic['url'] = i['web_url']
            news.append(dic)
    except KeyError:
        print("No response")
    return news

def get_articles(begin_date,end_date):
    '''
    This function accepts a year in string format (e.g.'1980')
    and a query (e.g.'Amnesty International') and it will 
    return a list of parsed articles (in dictionaries)
    for that year.
    '''
    calls = 0 # avoid api limits
    all_articles = []
    api = NYTapi.articleAPI(key)
    for i in range(0,100): #NYT limits pager to first 100 pages. But rarely will you find over 100 pages of results anyway.
        calls = calls + 1
        print("Call: ", calls)
        try:
            articles = api.search(fq = {'section_name': 'World', 'type_of_material': 'News', 'source':['The New York Times']}, begin_date = begin_date, end_date = end_date, page = str(i))
            articles = parse_articles(articles)
            all_articles = all_articles + articles
            time.sleep(2)
        except:
            print("Fail on {c}, try next key".format(c=calls))

    return all_articles

def get_URLs(begin_date = '20070701', end_date = '20180325'):
    """
    Args: 
            begin_date: default as the 2007/07/01
            end_date: the end of the date of time frame for searching news articles
                              default as 2018/03/25
    """
    URLs = []
    meta_data = get_articles(begin_date, end_date)
    for article in meta_data:
        URLs.append(article['url'])
    return URLs

def main():
    """
    output:
    { 'title': {
            'ptime': time
            'geo': loc
            'content': body
        }
    }
    """
    begin_date = sys.argv[1]
    end_date = sys.argv[2]
    print('Begin Date: ', begin_date)
    print('End Date', end_date)
    URL = get_URLs(begin_date, end_date)
    print("Total Articles to scrape: ", len(URL))

    articles = {}
    k = 0
    last_date = ""

    end_date = datetime.datetime.strptime(end_date, "%Y%m%d")
    begin_date = end_date + timedelta(days=1)
    end_date = end_date + timedelta(days=20)
    begin_date = datetime.datetime.strftime(begin_date, "%Y%m%d")
    end_date = datetime.datetime.strftime(end_date, "%Y%m%d")

    for url in URL:
        k = k + 1
        print("Currently processing article ", k)
        try:
            r = requests.get(url)
        except:
            print("Network Error, Last Date = ", last_date)
            break
        soup = BeautifulSoup(r.text, 'html.parser')
        news = {}
        title = ""
        text = ""
        for p in soup.find_all('p', attrs={'class':'story-body-text story-content'}):
            text += p.get_text()
        news['content'] = text
        for tag in soup.find_all("meta"):
            if tag.get("name", None) == "DISPLAYDATE":
                news['DATE'] = tag.get("content", None)
                last_date = news['DATE']
            if tag.get("name", None) == "hdl":
                news['headline'] = tag.get("content", None)
                title = news['headline']
                print(title)
            if tag.get("name", None) == "utime":
                news['utime'] = tag.get("content", None)
            if tag.get("name", None) == "ptime":
                news['ptime'] = tag.get("content", None)
            if tag.get("name", None) == 'geo':
                news['geo'] = tag.get("content", None)
        articles[title] = news
        time.sleep(2)

    # Save full articles
    with open('../../../data/news/full/nyt_'+str(begin_date)+'_'+str(end_date)+'.json', 'w') as outfile:
        json.dump(articles, outfile)

    #Save last scraped log
    with open('../../../data/news/scrape_log', 'a') as out:
        out.write("Last Date: " + last_date)
    print("Total articles: ", k)


if __name__ == '__main__':
	main()
