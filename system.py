#!/usr/bin/env python

"""

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
import numpy  as np
import pandas as pd
from module_scraper        import scraper as scraper
from module_classification import bug_embeddings as bm
from module_embeddings     import word_embeddings as wm
from module_embeddings     import stacktraces_embeddings as sm
from module_preprocessing  import vocabulary_processes as vp
from module_preprocessing  import descriptions_preprocessing as dp
from module_preprocessing  import stacktraces_preprocessing  as sp
from module_classification import classifiers as classifiers


if __name__ == "__main__" :
  
  # *** scraping  ***
  # github repository url
  query_url = f"https://github.com/cgeo/cgeo/issues?page=1&q=is%3Aissue+is%3Aclosed"
  #scraping the github repo
  scraper.initializing(query_url)

  # *** word embeddings ***
  # clean and split descriptions data into validation,test and training set
  dp.data_preprocessing()
  # create vocabulary and corpus for words
  vp.create_vocabulary_corpus('words',skip_window = 2, min_occurance = 3)
  wm.word_embeddings_creation(skip_window = 2, embedding_dim=64, num_sampled = 64, learning_rate=0.1)
  #wm.hyper_parameters_estimation()
  
  # *** graph embeddings ***
  sp.stacktraces_preprocessing()
  vp.create_vocabulary_corpus('funcs',skip_window = 2,skip_window_t=2, min_occurance=2,true_neigh=2,false_neigh=10,valid_w=20,valid_w2=40,test_w=30)
  sm.stack_embeddings_creation(skip_window=2,embedding_dim=8,num_sampled=32)
  #sm.hyper_parameters_estimation()

  # run classifiers 
  tags, df_tags, issues_embeddings = bm.bugs_preprocessing(use_words = True, use_stacks = True)
  classifiers.run_classifiers(tags, df_tags, issues_embeddings)