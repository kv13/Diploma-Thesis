import os
import re
import json
from random   import seed
from random   import randint
from datetime import datetime
from module_preprocessing import descriptions_preprocessing as dp

# load and clean stack traces
def load_stacktraces(st_traces_ls,dir_path):
    
    total_stack_traces = 0

    for fname in os.listdir(dir_path):
        with open(os.path.join(dir_path,fname)) as json_file:

            #####################################
            # print("working on file",fname,"\n")
            #####################################

            # load data
            data = json.load(json_file)

            for counter,issue in enumerate(data):
                
                dirty_stack_trace = issue['stack_trace']

                if dirty_stack_trace != []:
                    total_stack_traces += 1

                    ###############################################
                    #print("stack trace on issue",counter + 1,"\n")
                    ###############################################
                    
                    if len(dirty_stack_trace) > 1:
                        dirty_stack_trace_1 = ' '.join(dirty_stack_trace)
                        stack_trace = clean_stack_trace(dirty_stack_trace_1)
                    else:
                        stack_trace = clean_stack_trace(dirty_stack_trace[0])
                    
                    # keep only non empty stack traces
                    # with more than one function call 
                    if stack_trace != []:
                        if len(stack_trace)>1:
                            st_traces_ls.append(stack_trace)


# remove text from the stack trace 
# keep only the sequence of functions
# returns a list with the function calls
def clean_stack_trace(stack_trace):

    clean_stack_trace = []
    temp_stack        = stack_trace.split(" at ")[1:]
    to_find           = re.compile("[|,|<|>]|/|\|=")
    
    # find where each function ends and keep only the path
    for f in temp_stack:
        temp      = f.find(')')
        temp_file = f[0:temp]

        # check the punctuations in order to avoid anything else
        match_obj = to_find.search(temp_file)
        if match_obj == None:
            filename = find_filename(temp_file)
            if filename != '':
                clean_stack_trace.append(filename)
    
    return clean_stack_trace


# remove the name of the function and store  
# only the file which contains the function.  
# This is done by tracking fullstops
def find_filename(value):
    filename = ""
    words    = value.split("(")
    if len(words)>=2:
        parts = words[0].split(".")
        filename = ".".join(parts[0:-1])
    return filename


def split_dataset(st_traces_ls,valid_size,test_size):
    train_set = list()
    valid_set = list()
    test_set  = list()

    seed(datetime.now())

    for i in range(valid_size):
        temp = randint(0,len(st_traces_ls)-1)
        valid_set.append(st_traces_ls.pop(temp))

    for i in range(test_size):
        temp = randint(0,len(st_traces_ls)-1)
        test_set.append(st_traces_ls.pop(temp))

    train_set = [i for i in st_traces_ls]
    return train_set,valid_set,test_set

def stacktraces_preprocessing(dir_path = 'data'):
    
    # define necessary parameters
    st_traces_ls   = list()  

    # load all issues which contains stack traces
    load_stacktraces(st_traces_ls,dir_path)
    
    # split stack traces into
    # train, validation, test set
    valid_size = int(0.2*len(st_traces_ls))
    test_size  = int(0.1*len(st_traces_ls))
    
    train_set,validation_set,test_set = split_dataset(st_traces_ls,valid_size,test_size)

    # print some statistics #
    print("size of train set"     , len(train_set))
    print("size of validation set", len(validation_set))
    print("size of test set"      , len(test_set))

    # save sets
    dp.save_file(train_set     , 'outputs/c_s_train.json')
    dp.save_file(validation_set, 'outputs/c_s_validation.json')
    dp.save_file(test_set      , 'outputs/c_s_test.json')