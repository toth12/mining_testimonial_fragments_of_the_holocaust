"""Query a gensim synset model."""
import constants
import pdb
from gensim.models import Word2Vec
import argparse


def main(term, distance=5, topn=50):
    """Print the synonyms in a precomputed model.

    It loads the model and query it. Distance is defined in the trained model.

    Parameters
    ----------
    term : {string}
        Term the synonyms of which is searched.
    distance : {number}, optional
    topn : {number}, optional
        Number of synonyms (the default is 50)
    """
    model = Word2Vec.load(constants.OUTPUT_FOLDER + 'synsets_window_' +
                          str(distance))
    similar_terms = model.wv.most_similar(term, topn=topn)
    for i, similar_term in enumerate(similar_terms):
        print(str(i) + ': ' + similar_term[0] + ', ' + str(similar_term[1]))
    pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--distance", type=int)
    parser.add_argument("-t", "--term", type=str)
    parser.add_argument("-n", "--number", type=int)
    args = parser.parse_args()
    if not args.term:
        raise Exception
    if not args.number:
        args.number = 50
    if not args.distance:
        args.distance = 5
    main(args.term, args.distance, args.number)
