# import labriares
import os 
import json 
import string
from random import seed
from random import randint
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def save_file(item,path):
    with open(path,'w') as f:
        json.dump(item,f)

        
def load_data(descriptions,dir_path):

    counter        = 0
    counter_issues = 0

    for fname in os.listdir(dir_path):
        with open(os.path.join(dir_path,fname)) as json_file:

            ######################################
            counter += 1
            print(counter,": reading file",fname)
            ######################################

            # load data in json format
            data = json.load(json_file)
            for p in data:

                ######################################
                issue_name     = p['name']
                counter_issues += 1
                #print("  ",counter_issues,")",issue_name)
                ######################################

                issue_desc     = p['description']
                
                # add all non empty issues and non dublicate.
                if issue_desc != [] and issue_desc not in descriptions:
                    descriptions.append(issue_desc)


def clean_data(clean_descriptions,raw_descriptions):
    
    # define stop words
    all_stopwords = set(stopwords.words('english'))

    # define translator to translate punctuation to white space
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))

    for desc in raw_descriptions:

        # join all lines into one sentence
        sentence = ' '.join(desc)

        # translate punctuation
        new_sentence = sentence.translate(translator)

        # split the sentense in words
        words = new_sentence.split()
        words_sw = [w.lower() for w in words if not w.lower() in all_stopwords and len(w)>1]
        
        if words_sw != []:
            clean_descriptions.append(words_sw)


def stemming_data(descriptions):
    
    stemmer = PorterStemmer()
    
    for desc in descriptions:
        for counter in range(len(desc)):
            if desc[counter].isalpha():
                desc[counter] = stemmer.stem(desc[counter])


def split_dataset(descriptions,valid_size,test_size,min_size):
    
    valid_set = []
    test_set  = []
    
    # random select descriptions.
    seed(datetime.now())

    for i in range(valid_size):
        flag = False
        while flag == False:
            temp = randint(0,len(descriptions)-1)
            if len(descriptions[temp]) >= min_size:
                valid_set.append(descriptions.pop(temp))
                flag = True
    
    for i in range(test_size):
        flag = False
        while flag == False:
            temp = randint(0,len(descriptions)-1)
            if len(descriptions[temp]) >= min_size:
                test_set.append(descriptions.pop(temp))
                flag = True
    
    return valid_set,test_set


def data_preprocessing(dir_path = 'data'):
    
    # the first time the below command should run to download stopwords
    # nltk.download('stopwords')

    # define necessary parameters
    raw_descriptions = []
    min_size         = 10
    
    # load all issues descriptions
    load_data(raw_descriptions,dir_path)

    # split and clean descriptions
    clean_descriptions = []
    clean_data(clean_descriptions,raw_descriptions)
    
    # stemming, it's not necessary step. Comment out if you dont want to apply stemming
    stemming_data(clean_descriptions)

    # split data set to train,validation and test set
    # validation and test set would have 20% of total data.
    total_desc = len(clean_descriptions)
    valid_size = int(0.3  * total_desc)
    test_size  = int(0.1  * total_desc)

    valid_set,test_set = split_dataset(clean_descriptions,valid_size,test_size,min_size)
    
    # save validation set, test set and train set
    path_name = 'outputs'
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    
    save_file(test_set          , 'outputs/c_d_test.json')
    save_file(valid_set         , 'outputs/c_d_validation.json')
    save_file(clean_descriptions, 'outputs/c_d_train.json')
    
    # print some statistics #
    print("total unique descriptions", total_desc)
    print("size of train set"        , len(clean_descriptions))
    print("size of validation set"   , valid_size)
    print("size of test set"         , test_size)

