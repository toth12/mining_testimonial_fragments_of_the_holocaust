"""Print the topics underlying a given."""
from utils import gensim_utils, blacklab, text
from gensim.models.phrases import Phrases, Phraser
import string
from nltk.corpus import stopwords
import constants
import argparse
import inspect
import difflib
import numpy as np
import pdb
import matplotlib.pyplot as plt
from corextopic import vis_topic as vt
import json
from sklearn.ensemble import IsolationForest
from kneed import KneeLocator
from sklearn.manifold import TSNE
import pandas as pd
from plotly import express as px
import plotly
import numpy as np
from sklearn.decomposition import PCA


#python get_topic_model_concordance_corex.py -q '<s/> <s/> <s/> <s/>(<s/> containing ["\["|"\("] []{0,10} [lemma="cry"] []{0,10} ["\]" | "\)"])' -o numbLDA -w 0 -topicn 50

#parameter: no_below=10, no_above=0.75,

#python get_topic_model_concordance_corex.py -q '<s/> <s/>(<s/> containing ["\["|"\("] []{0,10} [lemma="cry"] []{0,10} ["\]" | "\)"])' -o numbLDA -w 0 -topicn 20

#paramater no_below=5, no_above=0.75, topicn - 20

#python get_topic_model_concordance_corex.py -q '["\["|"\("] []{0,10} [lemma="cry"] []{0,10} ["\]" | "\)"]' -w 50 -o numbLDA -topicn 20
#no_below=10, no_above=0.75,

#very good python get_topic_model_concordance_corex.py -q '["\["|"\("] []{0,10} [lemma="cry"] []{0,10} ["\]" | "\)"]' -w 50 -o numbLDA -topicn 20 
# parameter 10

def main(query, output_filename, terms_to_remove=[], window=0, topicn=50):
    """Query a corpus and find the topics underlying the concordance.

    Parameters
    ----------
    query : {string}
        Valid CQL query.
    output_filename : {string}
        Absolute path to the file where results are printed.
    terms_to_remove : {list}, optional
        List of those terms that are to be removed when creating the topic
        model (the default is [])
    window : {number}, optional
        Window of words used when creating the concordance (the default is 0)
    topicn : {number}, optional
        Number of topics (the default is 50)
    """
    print('Training lda model began')
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    query_parameters = [(i, values[i]) for i in args]
    document_collection_original = blacklab.search_blacklab(query,
                                                            window=window,
                                                            lemma=True,
                                                            include_match=True,right=False)

    # Eliminate overlapping results

    # Find those ids that are present more than once
    ids = {i['testimony_id']: []for i in document_collection_original}

    for element in document_collection_original:
        ids[element['testimony_id']].append(element)

    for element in ids:
        if len(ids[element]) > 0:
            for i in range(0, len(ids[element]) - 1):
                match_element_1 = ids[element][i]['complete_match']

        # Iterate through and try to find similar matches
                for item in ids[element][i + 1:]:
                    match_element_2 = item['complete_match']
                    diff = difflib.SequenceMatcher(None, match_element_1,
                                                   match_element_2).ratio()
                    if diff > 0.5:
                        try:
                            document_collection_original.pop(document_collection_original.index(item))
                        except:
                            pass
    # Eliminate search terms
    terms_to_remove = terms_to_remove + constants.DATA_SPECIFIC_STOP_WORDS
    # get rid of stop words
    document_collection_filtered_without_search_terms = []
    for item in document_collection_original:
        new_text = []
        for word in item['complete_match'].split():
            if (word.lower() not in terms_to_remove):
                new_text.append(word)
        document_collection_filtered_without_search_terms.append(' '.join(new_text))

    # use the phraser model
    phraser_model = Phraser(Phrases.load(constants.OUTPUT_FOLDER +
                            'phrase_model'))
    document_collection = [' '.join(phraser_model[match.strip().split()]) for
                           match in
                           document_collection_filtered_without_search_terms]
    ids = [match['testimony_id'] for match in document_collection_original]

    # get rid of stop words
    document_collection_filtered = []
    for item in document_collection:
        new_text = []
        for word in item.split():
            if (word.lower() not in set(stopwords.words('english')) and
                (word[0] in string.ascii_uppercase +
                 string.ascii_lowercase)):
                new_text.append(word)
        document_collection_filtered.append(' '.join(new_text))
    
    tf_idf = gensim_utils.train_tf_idf_model(document_collection_filtered,terms_to_remove,no_below=5,no_above=0.75)
    
    
    document_collection_original = [element['match_word'] for element in document_collection_original]
    '''for i in sorted(tf_idf['removed'], reverse=True):
        del document_collection_original[i]'''
    pdb.set_trace()
    pca = PCA(n_components=50)
    tsne = (TSNE().fit_transform(
                       (pca.fit_transform(tf_idf['document_term'])
                        )))
     

    principalDf = pd.DataFrame(data=tsne,
                           columns=['principal_component_1',
                                    'principal_component_2'])

    principalDf['text'] = document_collection_original


    # Plot the results of the principal component analysis without clusters
    fig = px.scatter(principalDf, x="principal_component_2",
                 y="principal_component_1",hover_data=["text"])

    plotly.offline.plot(fig, filename='tf_idf_tsney.html')
    

    


if __name__ == '__main__':

    # open the CSV file
    # for instance python get_topic_model_concordance.py -q '["numb"]' -o numbLDA -w 5

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--synset", type=str)
    parser.add_argument("-q", "--query", type=str)
    parser.add_argument("-w", "--window", type=int)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-topicn", type=int)

    # normalize the window parameter, 0 value as deafult is not good
    # parser.add_argument('window')
    args = parser.parse_args()

    if not args.output:
        raise Exception

    if not args.topicn:
        args.topicn = 50

    if not args.window:
        args.window = 0

    if args.synset:
        input_data = text.ReadCSVasDict('Data/Input/Synsets/synsets.csv')

        synonyms = [element['Synonyms'].strip() for element in input_data if
                    element['Covering_Term'] == args.synset]
        query = []

        for element in synonyms[0].split(','):
            if '_' in element:
                result = '('
                element = element.split('_')
                for e in element:
                    result = result + '[word="' + e + '"]'
                result = result + ')'
                query.append(result)
            else:
                query.append('[lemma="' + element.strip() + '"]')
        query = '|'.join(query)
        query = '[]{10}(' + query + ')([ ]{10})'
        main(query, args.output, synonyms[0].split(','), args.window,
             args.topicn)
    else:
        main(args.query, args.output, ['everything','pause','PAUSES','year_old','first','non-english','inaudible','sob'], args.window, args.topicn)
