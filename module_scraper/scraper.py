"""

Scraper Module for scraping github repositories. It's focused on scraping issues only.
The algorithm is looking for the name of the issue, possible tags, the description
and available stack trace. Also it is looking for who opened it or who closed it 
if the issue is open or solved respectively. 

"""

# import libraries
import os
import re
import bs4
import csv
import time
import requests
import json, codecs
from pprint import pprint
from github import Github
from bs4 import BeautifulSoup


# write results on .json file
def write_json(page_objects,page):
    
    path_name = 'data2'
    file = 'data_word_emb'+str(page)+'.json'

    if not os.path.exists(path_name):
        os.makedirs(path_name)
    
    with open(os.path.join(path_name,file),'a') as f:
        json.dump(page_objects,f)


def find_tags(tags,each_issue):
    for counter,temp in enumerate(each_issue.find_all('a')):
        tag = str(temp.string)
        if tag != "None":
            tags.append(tag)
    remove_special_character(tags)


def remove_special_character(text):
    for i in range(0,len(text)):
        text[i] = text[i].replace('\n',' ')
        text[i] = text[i].replace('\t',' ')
        text[i] = re.sub(r"<.*?>"," ",text[i])
        # maybe a better solution is <[^<>]+> which matches any character except < or > 
        # one or more times included inside < and >


# function to search the entire page for stack trace
def search_all_page(stack_trace,page):
    temp = page.find_all('div',class_='edit-comment-hide')
    for i in temp:
        temp_2    = i.find('td')
        temp_desc = []
        calculate_desc(temp_2,temp_desc,stack_trace,0)
        if stack_trace != []:
            break


def find_desc(each,base_url,description,stack_trace,is_bug):

    next_url     = base_url + each.find('a')['href']
    response     = requests.get(next_url)
    html_content = response.content
    dom          = BeautifulSoup(html_content,'html.parser')
    temp         = dom.find('div',class_ = 'edit-comment-hide')
    temp_2       = temp.find('td')

    # find description and the stack trace(if any)
    calculate_desc(temp_2,description,stack_trace,1)

    # search the entire page for a stack trace
    if stack_trace == [] and is_bug == True:
        search_all_page(stack_trace,dom)
    
    # find the person who closed the issue
    who_closed_it = "none"
    temp_3        = dom.find_all('div',class_='TimelineItem-body')
    for i in temp_3:
        if i.text.find('closed this')!=-1:
            if i.find('a',class_='author Link--primary text-bold') is None:
                continue
            who_closed_it = i.find('a',class_='author Link--primary text-bold').text
    
    # remove special characters from the description and stack trace
    remove_special_character(description)
    if stack_trace != []:
        remove_special_character(stack_trace)
    
    return who_closed_it


def calculate_desc(html_text,description,stack_trace,flag):

    html_content = html_text.contents
    length       = len(html_content)

    if length == 1:
        # some lines are empty. No need to save them
        if html_content[0] != []:
            str_temp = str(html_content[0])
            if str_temp.find('.java:')!=-1 or str_temp.find('java.')!=-1 or str_temp.find('AndroidRuntime:')!=-1:
                stack_trace.append(str(html_content[0]))
            else:
                if flag == 1:
                    if str(html_content[0]).startswith('<code>') != True:
                        description.append(str(html_content[0]))
    else:
        for i in range(length):
            if type(html_content[i]) is bs4.element.NavigableString:
                # avoid writting empty lines and html tags <br>
                if len(html_content[i])>4:
                    str_temp = str(html_content[i])
                    if str_temp.find('.java:')!=-1 or str_temp.find('java.')!=-1 or str_temp.find('AndroidRuntime:')!=-1:
                        stack_trace.append(str(html_content[i]))
                    elif flag == 1:
                        description.append(str(html_content[i]))
            elif type(html_content[i] is bs4.element.Tag):
                # call recursively the function till length is 1
                calculate_desc(html_content[i],description,stack_trace,flag)

# search all issues
def scraping_process(query_url,seconds=60):
    
    # define some variables
    total_issues  = 0
    total_traces  = 0
    total_bugs    = 0
    pages_counter = 0
    issues        = []
    base_url      = "https://github.com/"

    # authentication process in order to make more requests
    # comment out if there is no GITHUB_TOKEN enviroment variable
    token = os.getenv("GITHUB_TOKEN")
    headers = {'Authorization': f'token {token}'}

    response = requests.get(query_url, headers = headers)
    #response = requests.get(query_url)

    # loop through all pages
    while True:
        response_code = response.status_code
        if response_code != 200:
            raise Exception("Error Occured")
        else:
            html_content = response.content
            dom          = BeautifulSoup(html_content,'html.parser')

            # find all issues in every page
            all            = dom.findAll('div', class_='flex-auto min-width-0 p-2 pr-3 pr-md-2')
            page_objects   = []
            pages_counter += 1
            
            # real scraping begins
            # search all issues per page
            for each in all:
                # find tags and who opens the issue
                tags = []
                find_tags(tags,each)
                
                # flag that is activated if the issue is bug
                is_bug = False
                for i in tags:
                    if i == "Bug":
                        is_bug     = True
                        total_bugs = total_bugs+1
                
                # find description, stack trace and who closed the issue
                description   = []
                stack_trace   = []
                who_closed_it = find_desc(each,base_url,description,stack_trace,is_bug)

                total_issues = total_issues+1
                if stack_trace !=[]:
                    total_traces = total_traces +1
                
                # write dictionary
                if len(tags)>=1:
                    issue_object = {'name':tags[0],'tags':tags[1:-1],'opened_by':tags[len(tags)-1],
                    'description':description,'stack_trace':stack_trace,'closed_by':who_closed_it}

                page_objects.append(issue_object)

                print(total_issues,tags[0])

            # write issues by page 
            write_json(page_objects,pages_counter)
    
            # visit the next page if exists
            end = dom.findAll('a',class_='next_page')
            if end == []:
                break
            
            next_url = base_url + end[1]['href']
            response = requests.get(next_url)

            # Sleep for t=seconds seconds
            time.sleep(seconds)
        
    return total_issues,total_bugs,total_traces 


def initializing(query_url):

    total_issues,total_bugs,total_traces = scraping_process(query_url)


    # print some statistics #
    print("Total issues in the repo",       total_issues)
    print("Total bugs in the repo",         total_bugs)
    print("Total stack traces in the repo", total_traces)
