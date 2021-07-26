#!/usr/bin/env python

"""

*********************************************
*      author:Konstantinos Vergopoulos      *
*                 Jule,2021                 *
*********************************************


This is the main script which is responsible to run all the components for the current project.
First, the script runs the scraper which is responsible to gather the data from a github 
repository (make sure you provide the url of the github repo). Once the scraper has finished, it 
will run some statistics which will decide if the stack traces graph can be produced based on 
the occurance of each function call.

Then, based on the decision, it will run the machine learning algorithms sequentially in the following order
    ~ Word embedding's creation using skip-grams method. 
      # You can either provide the hype parameters or let the script make hype parameter tuning with respect to AUC metric   
    ~ Stack traces's embedding is exactly like word embeddings
    ~ Tag Classification task
      # Dummy Classifier
      # Logistic Regression Classifier
      # Neural Network Classifier   

"""
import argparse

from nltk.corpus.reader.chasen import test
import module_scraper.scraper as scraper
from module_embeddings import word_embeddings as wm
from module_preprocessing import vocabulary_processes as vp
from module_preprocessing import descriptions_preprocessing as dp


if __name__ == "__main__" :
  # still need a lot of work to be complete.
  # %%%%%%%%%%% create an easy command line interface %%%%%%%%%%%
  parser = argparse.ArgumentParser()
  parser.add_argument("-a", "--all",     action = "store_true", help = "run all modules")
  parser.add_argument("-s", "--scraper", type = str, help = "use the module scraper, provide the url as command line argument")
  parser.add_argument("-p", "--preprocessing", action = "store_true", help = "omit scraping and move on data pre-processing")
  parser.add_argument("-hw","--hyperparametertuning",action="store_true",help="hype parameter tuning for word embeddings model")
  args = parser.parse_args()
  # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  if args.all:
    # github repository url
    query_url = f"https://github.com/cgeo/cgeo/issues?page=1&q=is%3Aissue+is%3Aclosed"
    # scraping the github repo
    scraper.initializing(query_url)
    
    # clean and split data into validation,test and training set
    dp.data_preprocessing()
    
    # create vocabulary and corpus for words
    vp.vocabulary_processes.create_vocabulary_corpus('words')
  
  elif args.scraper:
    # run only the scraper with a custom url
    query_url = f"{args.scraper}"
    scraper.initializing(query_url)
  
  elif args.preprocessing:
    print("edw mpenei???")
    dp.data_preprocessing()
    vp.create_vocabulary_corpus('words')

  elif args.hyperparametertuning:
    wm.hyper_parameters_estimation()
  

