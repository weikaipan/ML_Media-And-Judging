# Wikileak Data

## wikileak_scrape.py
 This scrape the wikileak cables with given start_time and end_time. Then it generates folders by year and stores the wikileak data in .json files.

### Fields of json object
dict_keys(['title', 'text', 'Date:', 'Canonical ID:', 'Original Classification:', 'Current Classification:', 'Handling Restrictions', 'Character Count:', 'Executive Order:', 'Locator:', 'TAGS:', 'Concepts:', 'Enclosure:', 'Type:', 'Office Origin:', 'Archive Status:', 'From:', 'Markings:', 'To:'])

## wikileaksToPickle.py
This converts the json objects to dictionary and store it in pickle file. Data preprocessing as below:
1. Convert 'Date' (e.g. "1988 January 2, 12:58 (Saturday)") to python datetime object 
2. Parse the values in field of 'To'
3. Only keep the fields of 'Date', 'text', 'From', 'To' for each cable
4. Store the data in dictionary with {key: date , value: list of cable}, cable['From', 'To', 'text']
