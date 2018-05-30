from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import time         
import json
import os, errno


def get_urls(start_date, end_date):
    '''
    This function retrieves pointers to the URLs of individual cables,
    matching a dated search query to https://wikileaks.org/plusd/.
    
    Date format is YYYY-MM-DD.    
    '''
    target_urls = []
    headers = {'User-agent': 'Mozilla/5.0'}
    url = 'https://wikileaks.org/plusd/?qproject[]=ps&qproject[]=cc&qproject[]=fp&qproject[]=ee&qproject[]=cg&q=&qtfrom='+start_date+'&qtto='+end_date+'#result'

    scraped_page = requests.get(url)#,headers=headers)        # Call the page
    soup = BeautifulSoup(scraped_page.text, 'html.parser')  # Extract the HTML from page text
    table = soup.find("table", attrs={'id': 'doc_list'})    # Look for the table that contains the documnet list
    
    try:
        for row in table.findAll('tr')[1:]:                     # For each row in that table
            target_urls.append(row.find("a").get("href"))       # Add the link to our target_urls list
    except:
        pass
    return target_urls

def get_cable(url_pointer, full_text=False, save=False, save_fpath = '', return_json=True):
    '''
    This function scrapes the URL of a cable for its content.
    
    url_pointer     specifies URL
    full_text       TRUE = retrieve body of cable, FALSE = retrieve metadata only
    save            TRUE = save as TXT file, FALSE = don't save
    return_json     TRUE = return JSON object, FALSE = don't return anything
    save_fpath      specifies folder in which to save the cable
    '''
    
    headers = {'User-agent': 'Mozilla/5.0'}
    url = 'https://wikileaks.org'+url_pointer
    
    scraped_page = requests.get(url,headers=headers)        # Call the page
    soup = BeautifulSoup(scraped_page.text, 'html.parser')  # Extract the HTML from page text
#    print("soup:", soup)
    table = soup.find("table", attrs={'id': 'synopsis'})    # Look for the table that contains the synopsis
    cable = {}
#    print("getin")
    cable["title"] = table.find("td").text                  # Store cable title
    
    if full_text==True:                                     # Store cable text
        cable["text"] = soup.find("div", attrs={"id":"tagged-text"}).text.replace("\n", " ").replace("  ", " ")
#        print(cable["text"])

    for row in table.findAll("tr"):                        # Loop through table rows
        for cell in table.findAll("td"):                   # Loop through table columns
            if cell.find("a")!=None:
                key = cell.find("div", attrs={'class':'s_key'}).text
                value = cell.find("div", attrs={'class':'s_val'}).text
                
                cable[key]=value                           # Store table metadata
    cc = {}    
    if save == True:
        # Save cable to file
        with open(save_fpath+cable['Canonical ID:']+'.json', 'w') as outfile:
            json.dump(cable, outfile)

    if return_json == True:                                # Return cable
        cc['title'] = cable['title']
        cc['text'] = cable['text']
#        print(cc['text'])
        return cc

def save_cables_to_file(start_date, end_date, saveFilePath):
    '''
    This retrieves cables containing the metadata from cables in the date range.
    It also saves the individual files to disk.
    
    Dependencies: get_urls, get_cable
    '''
    target_urls =  get_urls(start_date, end_date)   # Get URLS matching the target date range
    cables = []
    for url in target_urls:                # Loop through all URLS
        try:
            cable = get_cable(url, full_text=True, save=True, save_fpath=saveFilePath) # Retrieve cable contents
            cables.append(cable)               # Add dictionary to list 'cables'
        except:
            print("Failed on", url)
    return cables


def main():
    daterange = [str(m) for m in pd.date_range('1987-01-01','1987-12-31' , freq='1M')]
    directory = ""
    count = 0
    for month in daterange:
        count+=1
        start_date = month[:8] + "01"
        end_date = month[:10]
        
        # Make new directory by year #
        if count % 12 == 1:
            directory = month[:4]
            print(directory)
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        elif count % 12 == 0:
            count = 0
        

        # Scrape data and write into file
        cb = save_cables_to_file(start_date, end_date, directory+'/')
#        print(cb)

        time.sleep(1)

if __name__ == '__main__':
	main()
