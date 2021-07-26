import os
import json
import pickle
import collections
import numpy as np


def load_dataset(mode,file_name):
    dir_path = 'outputs'
    if mode == 'words':
        file_name = 'c_d_'+file_name
        with open(os.path.join(dir_path,file_name),'r') as json_file:
            custom_set = json.load(json_file)
            return custom_set
    elif mode =='funcs':
        pass
    else:
        raise Exception("Error Occured")


def save_corpus(corpus,mode):
    # make sure the directory exists
    directory = "outputs"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = directory+'/'+mode + "_corpus.txt" 
    np.savetxt(file_name,corpus,fmt="%d")


def load_corpus(mode):
    file_name = 'outputs/' + mode + '_corpus.txt'
    corpus = np.loadtxt(file_name,dtype ='int')
    return corpus


def save_vocabulary(w_dict,mode):
    # make sure the directory exists
    directory = "outputs"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = mode + "_vocabulary.txt"   
    with open(os.path.join(directory,file_name),"w") as file:
        for key in w_dict:
            file.write("%s, %s \n"%(key,str(w_dict[key])))


def load_vocabulary(mode):
    
    file_name = 'outputs/' + mode + '_vocabulary.txt'
    temp_dict = dict()
    
    with open(file_name) as file:
        lines = file.readlines()
        for line in lines:
            temp   = str(line)
            values = temp.split(',')
            temp_dict[values[0]] = int(values[1].replace("\n",""))
    
    return temp_dict


def save_test_pairs(test_dict,mode,file_name):
    dir_path  = 'outputs'
    file_name = dir_path + '/' + mode + '_' + file_name +'.pkl'
    with open(file_name,'wb') as file:
        pickle.dump(test_dict,file,pickle.HIGHEST_PROTOCOL)


def load_test_pairs(mode,file_name): 
    # unpickling test dictionary
    dir_path  = 'outputs'
    file_name = dir_path + '/' + mode + '_' + file_name
    with open(file_name,'rb') as infile:
        testing_dict = pickle.load(infile)
    return testing_dict


def create_testing_dict(test_set,min_occurance,num_words,num_words2,true_neigh,false_neigh,skip_window=2):
    
    # numerate all words in the dataset.
    temp_list = [value for item in test_set for value in item]
    count = []
    count.extend(collections.Counter(temp_list).most_common())
    
    # list temp_sentences now is useless
    del temp_list
    
    # remove rare words
    count[:] = [e for e in count if e[1]>=min_occurance]
    indexes  = [i for i in range(len(count)) if count[i][0] != -2]
    
    # split validation set into two sets one small used for cross entropy computation
    # and the other at the end to meassure results.
    if num_words2>0:
        
        samples2  = np.random.choice(indexes,num_words2,replace = False)
        target_w2 = [count[i][0] for i in samples2]
        w_dict2   = create_testing_pairs(test_set,count,target_w2,indexes,skip_window,true_neigh,false_neigh)
        
        # test on the "num_words" most frequent words
        tmp_indexes = [i for i in indexes if i not in samples2]
        target_w    = [count[tmp_indexes[i]][0] for i in range(num_words)]
        w_dict      = create_testing_pairs(test_set,count,target_w,indexes,skip_window,true_neigh,false_neigh)
        return w_dict2,w_dict
    
    else:
        # test on the "num_words" most frequent words
        target_w = [count[indexes[i]][0] for i in range(num_words)]
        w_dict   = create_testing_pairs(test_set,count,target_w,indexes,skip_window,true_neigh,false_neigh)
        return None,w_dict


def create_testing_pairs(test_set,count,target_w,indexes,skip_window,true_neigh,false_neigh):
    
    # initialize temporary buffer
    span   = skip_window*2+1
    buffer = collections.deque(maxlen = span)
    
    # initialize dictionary
    w_dict   = dict([(key, [[],[]]) for key in target_w])
    
    # find true neighbors for target words
    for desc in test_set:
        for w in target_w:
            temp_idx = [i for i,e in enumerate(desc) if w == e]
            for idx in temp_idx:
                find_context_words(desc,idx,skip_window,span,buffer)
                for i in range(1,len(buffer)):
                    if w_dict[w][0] == []:
                        w_dict[w][0].append(buffer[i])
                    elif buffer[i] not in w_dict[w][0]:
                        w_dict[w][0].append(buffer[i])
    
    # find false neigbors for target words
    for key in w_dict:
        neig_counter = 0
        flag         = True
        while flag  == True:
            random_idx   = np.random.choice(indexes,2*false_neigh,replace = False)
            for idx in random_idx:
                if count[idx][0] == key:
                    continue
                elif count[idx][0] in w_dict[key][0]:
                    continue
                elif count[idx][0] not in w_dict[key][1]:
                    w_dict[key][1].append(count[idx][0])
                    neig_counter += 1
                    if neig_counter >= false_neigh:
                        flag = False
                        break
    
    # choose randomly only true_neigh neighbors.
    removed_keys = []
    for key in w_dict:
        if len(w_dict[key][0])>=true_neigh:
            idx_neigh =  np.random.choice([i for i in range(len(w_dict[key][0]))],true_neigh,replace = False)
            w_dict[key][0] = [w_dict[key][0][i] for i in idx_neigh]
        else:
            removed_keys.append(key)
            
    if removed_keys != []:
        for key in removed_keys:
            w_dict.pop(key)

    return w_dict

 
def create_corpus(train_set,skip_window):
    
    # find total instances
    total_w = 0
    for func in train_set:
        total_w += len(func)
        
    # initialize the corpus which will keep all pairs
    max_size = total_w*2*skip_window
    corpus = -1*np.ones((max_size,2), dtype=np.int32)
    
    # initialize pointers for the iterations
    d_pointer  = 0
    w_pointer  = 0
    counter    = 0
    
    #initialize temporary buffer
    span   = 2*skip_window+1 
    buffer = collections.deque(maxlen = span)
    
    while counter< max_size:
        
        # avoid tags with -2
        while train_set[d_pointer][w_pointer] < 0:
            w_pointer += 1
            if w_pointer > len(train_set[d_pointer])-1:
                w_pointer  = 0
                d_pointer += 1
                if d_pointer > len(train_set) -1:
                    break
        
        # check if all issues have been analyzed
        if d_pointer > len(train_set)-1:
            break
        
        find_context_words(train_set[d_pointer],w_pointer,skip_window,span,buffer)
        
        for i in range(1,len(buffer)):
            corpus[counter][0] = buffer[0]
            corpus[counter][1] = buffer[i]
            counter += 1
            
        buffer.clear()
        
        if w_pointer == len(train_set[d_pointer]) -1:
            w_pointer  = 0
            d_pointer +=1
            if d_pointer > len(train_set) -1:
                break
        else:
            w_pointer += 1
    
    return corpus[0:counter].copy()


def find_context_words(description,w_index,skip_window,span,grams_list):
    
    # the target word in the first place
    grams_list.append(description[w_index])
    
    # initialize two pointers
    counter = 1
    data_index = w_index-1
    
    while counter < span:
        # look left from target word
        if counter<=skip_window:
            # if data_index<0 => out of bound no more words to take into account
            if data_index < 0:
                data_index = w_index  + 1
                counter    = skip_window + 1
            # if the word is not in the dict skip it
            elif description[data_index] == -2:
                data_index -= 1
            else:
                grams_list.append(description[data_index])
                counter    += 1
                data_index -= 1
                if counter > skip_window:
                    data_index = w_index + 1
        # look right from target word
        else:
            if data_index >= len(description):
                counter = span + 1
            elif description[data_index] == -2:
                data_index += 1
            else:
                grams_list.append(description[data_index])
                counter    += 1
                data_index += 1    


def create_vocabulary(train_set, validation_set, test_set, mode, min_occurance,unk_item):
    
    # create vocabulary based on the frequency of each word or function call.
    # remove rare words, which occurs less time than min_occurance from voc
    # word2id:  dictionary which contains the vocabulary and it's int id for words
    # func2id:  dictionary which contains the vocabulary and it's int id for each function call

    #find frequency for each word/function call
    temp_list = [value for item in train_set for value in item]

    count     = []
    count.extend(collections.Counter(temp_list).most_common())
    count[:]  = [e for e in count if e[1]>=min_occurance]

    # create vocabulary
    #vocabulary_size = len(count)

    # assign an id to each function
    f_w2id = dict()
    f_w2id[unk_item] = -2

    for i,(item,_) in enumerate(count):
        f_w2id[item] = i
    
    # list count now is useless
    del count

    train_set_id = [[f_w2id.get(func,-2) for func in i] for i in train_set]
    valid_set_id = [[f_w2id.get(func,-2) for func in i] for i in validation_set]
    test_set_id  = [[f_w2id.get(func,-2) for func in i] for i in test_set ]

    return train_set_id, valid_set_id, test_set_id,f_w2id

    

def create_vocabulary_corpus(mode, skip_window=1, min_occurance=1, unk_item = "UNK", 
                            true_neigh = 8, false_neigh = 30, valid_w=80,valid_w2=70,test_w=100):

    # load train, validation and testing set
    train_set      = list()
    validation_set = list()
    test_set       = list()

    train_set      = load_dataset(mode, 'train.json')
    validation_set = load_dataset(mode, 'validation.json')
    test_set       = load_dataset(mode, 'test.json')
    
    # create vocabulary 
    train_set_id,valid_set_id,test_set_id,vocab_dict = create_vocabulary(train_set, validation_set, test_set, mode, min_occurance,unk_item)
    
    # save the vocabulary
    save_vocabulary(vocab_dict,mode)

    # create corpus with training pairs
    corpus         = create_corpus(train_set_id,skip_window)

    # train_set_id now is useless
    del train_set_id

    # save them 
    save_corpus(corpus,mode)

    # create testing pairs
    # for stack traces test pairs min occurance for each function call is hardcoded at 2, 
    # because the dataset is small.
    if mode == 'words':
        min_occur = min_occurance
    elif mode == 'funcs':
        min_occur = 2
    
    _,test_dict  = create_testing_dict(test_set_id,min_occur,test_w,0,true_neigh,false_neigh)
    save_test_pairs(test_dict,mode,'testing_pairs')
    del test_dict

    # create validation pairs
    v_dict2,v_dict = create_testing_dict(valid_set_id,min_occur,valid_w,valid_w2,true_neigh,false_neigh)
    
    save_test_pairs(v_dict , mode, 'validation_pairs')
    save_test_pairs(v_dict2, mode, 'validation2_pairs')
