import os
import json
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import module_preprocessing as pre

def compute_embeddings(arithmetic_descriptions,arithmetic_stack_traces,word_embedding_matrix,\
                       stack_embedding_matrix,use_words,use_stacks):

    total_embeddings_dim = 0
    descriptions_dim     = 0
    stack_traces_dim     = 0
    
    if use_words == True:
        descriptions_dim     = np.shape(word_embedding_matrix)[1]
        total_embeddings_dim = total_embeddings_dim + descriptions_dim
        
    if use_stacks == True:
        stack_traces_dim     = np.shape(stack_embedding_matrix)[1]
        total_embeddings_dim = total_embeddings_dim + stack_traces_dim
    
    # make sure that in any case there are something to compute
    if total_embeddings_dim ==0:
        return None

    num_issues        = len(arithmetic_descriptions)
    issues_embeddings = np.zeros((num_issues,total_embeddings_dim))
    
    # compute arithmetic representation for each bug
    for counter in range(len(arithmetic_descriptions)):
        
        temp_desc   = arithmetic_descriptions[counter]
        temp_stack  = arithmetic_stack_traces[counter]
        total_words = 0
        total_funcs = 0
        
        if use_words == True:
            for word in temp_desc:
                if word != -2:
                    total_words += 1
                    issues_embeddings[counter][0:descriptions_dim] = issues_embeddings[counter][0:descriptions_dim] + word_embedding_matrix[word]
            if total_words != 0 :
                issues_embeddings[counter]    /= total_words
        
        
        if use_stacks == True:
            for func in temp_stack:
                if func != -2:
                    issues_embeddings[counter][descriptions_dim:] = issues_embeddings[counter][descriptions_dim:] + stack_embedding_matrix[func]
                    total_funcs += 1
            if total_funcs != 0:
                issues_embeddings[counter][descriptions_dim:] = issues_embeddings[counter][descriptions_dim:] / total_funcs 
            
    return issues_embeddings  

def load_issues(tag_labels, descriptions, stack_traces, dir_path = 'data'):
    
    for fname in os.listdir(dir_path):
        with open(os.path.join(dir_path,fname)) as json_file:
            
            data = json.load(json_file)
            for issue in data:
                
                tags = issue['tags']
                for i in range(len(tags)):
                    tags[i] = tags[i].strip()
                
                description = issue['description']
                stack_trace = issue['stack_trace']

                if tags != [] and stack_trace !=[] and description != []: #(description != [] or stack_trace != []):
                    tag_labels.append(tags)
                    descriptions.append(description)
                    stack_traces.append(stack_trace)
    
def clean_description(description):
    
    # define stop words
    all_stopwords = set(stopwords.words('english'))
    
    #define translator to translate punctuation to white space
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    
    #join all lines into one sentence
    sentence     = ' '.join(description)
    
    #translate punctuation
    new_sentence = sentence.translate(translator)
    
    #split the sentense in words
    words = new_sentence.split()
    
    words_sw = [w.lower() for w in words if not w.lower() in all_stopwords and len(w)>1]
    
    return words_sw

def clean_data(descriptions, stack_traces, use_stemming):
    
    clean_descriptions = list()
    clean_stack_traces = list()
    
    for i in range(len(descriptions)):
        
        temp_desc   = descriptions[i]
        temp_trace  = stack_traces[i]
        stack_trace = []
        clean_desc  = []
        
        if temp_trace != []:
            if len(temp_trace)>1:
                stack_trace = pre.stacktraces_preprocessing.clean_stack_trace(' '.join(temp_trace))
            else:
                stack_trace = pre.stacktraces_preprocessing.clean_stack_trace(temp_trace[0])
            
        if temp_desc  != []:
            clean_desc = clean_description(temp_desc)
            
        clean_descriptions.append(clean_desc)
        clean_stack_traces.append(stack_trace)
            
    if use_stemming == True:
        pre.descriptions_preprocessing.stemming_data(clean_descriptions)
        
    return clean_descriptions,clean_stack_traces

def bugs_preprocessing(use_words, use_stacks,use_stemming = True):

    # load word embeddings
    word_embedding_matrix = np.loadtxt('../results/word_embeddings.txt', dtype=np.float64)

    # load stack traces embeddings 
    stack_embedding_matrix = np.loadtxt('../results/funcs_embeddings.txt', dtype=np.float64)

    # load vocabularies

    word2id = pre.vocabulary_processes.load_vocabulary('words')
    func2id = pre.vocabulary_processes.load_vocabulary('funcs')

    #load tags and descriptions
    tag_labels   = list()
    descriptions = list()
    stack_traces = list()

    # load issues
    load_issues(tag_labels,descriptions,stack_traces)

    # transform data to arithmetic representation
    clean_descriptions,clean_stack_traces = clean_data(descriptions,stack_traces,use_stemming)

    clean_descriptions_2 = list()
    clean_stack_traces_2 = list()
    clean_tags_2         = list()

    # remove empty stack traces or dublicate issues
    for counter in range(len(clean_stack_traces)):
        
        if clean_stack_traces[counter] != []:
            
            flag   = False
            flag_2 = False 
            
            # remove empty stack traces 
            for i in clean_stack_traces[counter]:
                func = func2id.get(i,-2)
                if func != -2:
                    flag_2 = True
                    break
            if flag_2 == False:
                continue
            
            # check for dublicates
            for counter_2 in range(len(clean_stack_traces_2)):
                if clean_descriptions[counter] == clean_descriptions_2[counter_2] and \
                    clean_stack_traces[counter] == clean_stack_traces_2[counter_2]:
                        flag = True
                        break
            
            if flag == False:
                clean_stack_traces_2.append(clean_stack_traces[counter])
                clean_descriptions_2.append(clean_descriptions[counter])
                clean_tags_2.append(tag_labels[counter])
                        
    del clean_descriptions
    del clean_stack_traces

    del descriptions
    del stack_traces

    #arithmetic_transformations
    arithmetic_descriptions = [[word2id.get(word,-2) for word in desc]   for desc in clean_descriptions_2]
    arithmetic_stack_traces = [[func2id.get(func,-2) for func in trace] for trace in clean_stack_traces_2]

    del clean_descriptions_2
    del clean_stack_traces_2

    issues_embeddings  = compute_embeddings(arithmetic_descriptions,arithmetic_stack_traces,\
        word_embedding_matrix,stack_embedding_matrix, use_words ,use_stacks)
    
    tag_labels = list()
    # copy by reference in order to avoid to change every where the variable name
    tag_labels = clean_tags_2

    # define which tags will be used for classification
    tags = ['Bug','Google Play or Beta feedback','Prio - High']
    #tags = ['>test-failure','Team:Distributed','>bug',':Distributed/Snapshot/Restore']
    #tags = ['type: bug','for: stackoverflow','status: invalid','for: external-project']
    
    no_tags = 4
    np_tags = np.zeros((len(arithmetic_descriptions),no_tags))

    for counter in range(len(tag_labels)):
        for counter_2,value in enumerate(tags):
            if value in tag_labels[counter]:
                np_tags[counter][counter_2] = 1
                
    df_tags = pd.DataFrame(np_tags, columns = tags)                    

    return tags, df_tags, issues_embeddings