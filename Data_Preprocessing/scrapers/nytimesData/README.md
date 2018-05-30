# Description

This is a scraper for New York Times World section news.

## Requirments

* NYTapi
* BeautifulSoup 

## Scripts

* nytimes_scraper.py:
The main scraper that receives two queries: 1) begin_date 2) end_date, then scrapes news metadata and news body using Beautiful Soup and request library.

* NYTapi.py
An open source python library that serves as the backend of ```nytimes_scraper.py```.

* apikeys.py
Put apikeys in this script query metadata of news.

## Output Format

The generated news format is a dictionary in python and out put as ```.json```
```
{ 'title': {
            'ptime': time
            'geo': loc
            'content': body
        }
    }
```

## Output Directory

The output file is default as below:
```
Media-And-Judging/data/news/full/nyt_[begin_date]_[end_date].json
```



## Acknowledgements

(New York Times Public Api Specs)[https://github.com/NYTimes/public_api_specs]
(get-nytimes-articles)[https://github.com/casmlab/get-nytimes-articles]




