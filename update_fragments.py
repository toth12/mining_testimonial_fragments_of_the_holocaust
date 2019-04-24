"""Find the sentence id of fragments."""
# -*- coding: utf-8 -*-

import pandas
import constants
import pdb
import requests
import json
from utils import blacklab, db, text
import math


def read_csv(filename):
    df = pandas.read_excel(filename, encoding='utf-8')
    return df.T.to_dict().values()


def find_sentence_id(fragments):
    updated_fragments = []
    for fragment in fragments:
        if not math.isnan(fragment['end_sentence_index']):
            updated_fragments.append(fragment)
        else:
            label = fragment['label']
            # tokenize with Stanford Parser
            props = {'annotators': 'tokenize'}

            # set the encoding of the annotator
            requests.encoding = 'utf-8'
            # make a request
            r = requests.post('http://localhost:9000/', params={'properties':
                              json.dumps(props)},
                              data=label.encode('utf-8'))
            result = json.loads(r.text, encoding='utf-8')
            query = []

            for i, token in enumerate(result['tokens']):

                if ('...'in token['word'] and ((i == 0) or
                   i == len(result['tokens']) - 1)):
                    continue
                elif ('...'in token['word']):
                    query.append('[]{0,50}')
                elif ('-'in token['word']):
                    query.append('[]{0,3}')
                elif ("n't"in token['word']):
                    query.append('[]')
                elif ("'re"in token['word']):
                    query.append('[]')
                elif ("?"in token['word']):
                    query.append('[]')
                elif ("."in token['word']):
                    query.append('[]')
                elif ("'s"in token['word']):
                    query.append('[]')
                else:
                    query.append('["' + token['word'] + '"]')

            query = ' '.join(query)
            try:
                sentence = blacklab.search_blacklab(query, window=0,
                                                    lemma=False,
                                                    include_match=True,
                                                    document_id=fragment['testimony_id'])
                token_end = sentence[0]['token_end']
                token_start = sentence[0]['token_start']
                mongo = db.get_db()
                results = mongo.tokens.find({'testimony_id':
                                            fragment['testimony_id']},
                                            {'_id': 0})
                tokens = list(results)[0]['tokens']
                sentenceStart = tokens[token_start]['sentence_index']
                sentenceEnd = tokens[token_end]['sentence_index']
                originalsentence = sentence[0]['complete_match']
                fragment['original_sentences'] = originalsentence
                fragment['end_sentence_index'] = sentenceEnd
                fragment['start_sentence_index'] = sentenceStart
                updated_fragments.append(fragment)
            except:
                print("The following query returned a null result")
                print(query)
                pdb.set_trace()
            if len(sentence) == 0:
                print('The following fragment could not be found')
                print(query)
                print(label)
                tokenized_fragment = [token['word'] for token in
                                      result['tokens']]
                print(tokenized_fragment)
                print('\n')
                pdb.set_trace()

    return updated_fragments


if __name__ == '__main__':
    fragments = read_csv(constants.INPUT_FOLDER + '/Fragments/' +
                         'testimonial_fragments.xlsx')
    fragments = list(fragments)

    old_fragments = read_csv(constants.INPUT_FOLDER + '/Fragments/' +
                             'testimonial_fragments_old.xlsx')
    old_fragments = list(old_fragments)
    updated_fragments = find_sentence_id(fragments)

    new_fragments = updated_fragments + old_fragments
    text.write_to_csv(new_fragments,
                      constants.OUTPUT_FOLDER +
                      'testimonial_fragments_updated.csv')
