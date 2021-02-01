from utils import gensim_utils, blacklab, text
import constants
import pdb
from gensim.models import TfidfModel
from gensim import matutils as mu
import numpy as np
from corextopic import corextopic as ct
from corextopic import vis_topic as vt
from pymongo import MongoClient
import random
import matplotlib.pyplot as plt
import fasttext






def query(db_name,collection_name,query,projection): 
    """Takes a db name, collection name, query and projection as input, and returns the result as a list"""
    client = MongoClient()
    client = MongoClient('localhost', 27017)
    db = client[db_name]
    collection = db[collection_name]
    #        pdb.set_trace()
    result = []
    for results in collection.find(query,projection):
        result.append(results)
    client.close()  
    return result

def chunks(li, n):
    """Returns every n elements of a list."""
    li = [element for element in li if len(element) > 0]
    # For item i in a range that is a length of l,
    
    for i in range(0, len(li), n):
        result = []
        # Create an index range for l of n items:
        [result.extend(element) for element in li[i:i + n]]
        if "cry" in result:
            new_result = []
            for element in result:
                if element !="cry":
                    new_result.append(element)
            res = '__label__cry '+ ' '.join(new_result)
            yield res
        else:
            
            yield '__label__not_cry '+ ' '.join(result)

import resource

all_words = gensim_utils.load_gensim_dictionary_model("Data/Output/gensimdictionary_all_words_with_phrases_filtered_no_below_10_no_above_095")
all_words = list(all_words.token2id.keys())

ids = text.read_json(constants.INPUT_FOLDER + 'testimony_ids.json')
metadata = []
all_bows = []
complete_vocab = []

# Get all sentences
for i, element in enumerate(ids):
    gender = query('lts','testimonies',{'testimony_id':element['testimony_id']},{'gender':1})
    gender = gender[0]['gender']
    if len(gender) ==0:
        continue
    results = blacklab.iterable_results('<s/>',
                                            lemma=True,
                                            document_ids=[element
                                                            ['testimony_id']],
                                            window=0)



    print (i)
    print ('\n')
    document_bows=chunks(results,8)

    if i == 0:
        with open('fastext_data.txt', 'w') as f:
        
            
            for element in document_bows:
            
                f.write("%s\n" % element)
            #all_bows.append(element)
            metadata.append({'gender':gender})
            #complete_vocab.extend(element)
        #complete_vocab = list(set(complete_vocab))
    else:
        with open('fastext_data.txt', 'a') as f:
        
            
            for element in document_bows:
            
                f.write("%s\n" % element)
            #all_bows.append(element)
            metadata.append({'gender':gender})

    print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

print ('collecting data finished')


model = fasttext.train_supervised('fastext_data.txt')

pdb.set_trace()
