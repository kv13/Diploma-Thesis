#!/usr/bin/env python

"""

*********************************************
*      author:Konstantinos Vergopoulos      *
*                 Jule,2021                 *
*********************************************


This is the main script which is responsible to run all the components for the current project.
First, the script runs the scraper which is responsible to gather the data from a github repository, make sure you provide the url of the github repo.
Once the scraper has finished will run some statistics which will decide if the stack traces graph can be produced based on the occurance of each function call.
Then based on the decision will run the machine learning algorithms sequentially in the following order
    ~ Word embedding's creation using skip-grams method. 
      # You can either provide the hype parameters or let the script make hype parameter tuning with respect to AUC metric   
    ~ Stack traces's embedding's exactly like word embeddings
    ~ Tag Classification task
      # Dummy Classifier
      # Logistic Regression Classifier
      # Neural Network Classifier   

"""

import src.module_scraper as scraper

if __name__ == "__main__" :

    # github repository url
    query_url = f"https://github.com/cgeo/cgeo/issues?page=1&q=is%3Aissue+is%3Aclosed"
    scraper.main(query_url)