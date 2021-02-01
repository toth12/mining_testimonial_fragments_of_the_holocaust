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




def query(db_name,collection_name,query,projection,complete_doc = False): 
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
        yield result

import resource

ids = text.read_json(constants.INPUT_FOLDER + 'testimony_ids.json')
metadata = []
all_bows = []
complete_vocab = []
# Get all sentences
def get_all_sentences(ids,complete_doc=False):
    result = []
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
        if complete_doc ==  True:
            document_bows = [element for sentence in results for element in sentence]
        else:
            document_bows=chunks(results,3)


        if complete_doc ==  True:
            yield document_bows
        else:
            for element in document_bows:
                element = [token.lower() for token in element]
                
                metadata.append({'gender':gender})
                yield element
            #complete_vocab.extend(element)
        #complete_vocab = list(set(complete_vocab))

    #print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


sentences = get_all_sentences(ids,complete_doc=True)

from gensim.corpora import Dictionary
dct = Dictionary()
for element in sentences:

    
    gensim_utils.add_documents_to_gensim_dictionary(dct, [element])

    

dct.filter_extremes(no_below=50, no_above=0.8)

sentences = get_all_sentences(ids)
gensim_corpus = [dct.doc2bow(sentence) for sentence in sentences]

model_tfid = TfidfModel(gensim_corpus,dictionary = dct,)
corpus_tfidf = model_tfid[gensim_corpus]



'''
print ('collecting data finished')

cry_events = random.sample(range(1, len(metadata)),100)

for element in cry_events:
    metadata[element]['cry'] = True



dct = gensim_utils.initialize_gensim_dictionary([complete_vocab])

gensim_utils.add_documents_to_gensim_dictionary(dct, all_bows)
dct.filter_extremes(no_below=50, no_above=0.8)



gensim_corpus = [dct.doc2bow(bag_of_word) for bag_of_word in all_bows]
model_tfid = TfidfModel(gensim_corpus,dictionary = dct,)
corpus_tfidf = model_tfid[gensim_corpus]
'''
features = list(dct.token2id.keys())
document_term=mu.corpus2dense(corpus_tfidf, len(features)).T
document_term = np.where(document_term>0, 1, 0)



print (len(features))
print ('gensim finished')


# Run Corex topic modelling
topic_model = ct.Corex(n_hidden=20, max_iter=200, verbose=False, seed=8)
topic_model.fit(np.matrix(document_term), words=dct.token2id.keys())

# Print document topic matrix: topic_model.log_p_y_given_x

# Print the key topics
topics = topic_model.get_topics()
topics_to_print = []
for n,topic in enumerate(topics):
    topic_words,_ = zip(*topic)
    topics_words_values = []
    for element in topic:
        topics_words_values.append(element[0] + ' (' +
                                   str(np.round(element[1], decimals=3)) + ')')
    topics_to_print.append(','.join(topics_words_values))
    print('{}: '.format(n) + ','.join(topic_words))


# Print top ten documents of each topic

top_docs = topic_model.get_top_docs(n_docs=10, sort_by='log_prob')
selected_chunked_bows = []
for topic_n, topic_docs in enumerate(top_docs):
    docs, probs = zip(*topic_docs)
    docs = [str(element) for element in docs]
    topic_str = str(topic_n + 1) + ': ' + ','.join(docs)
    doc_bows = []
    '''for doc in docs:
        doc_bows.append(' '.join(all_bows[int(doc)]))'''
    print(topic_str)

features = list(dct.token2id.keys())


# Create a report on the topic modelling (folder named topic-model-report created automatically)
vt.vis_rep(topic_model, column_label=features, prefix='topic-model-report')

pdb.set_trace()