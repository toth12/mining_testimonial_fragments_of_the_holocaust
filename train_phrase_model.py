"""Train gensim synset model."""
from utils import gensim_utils, blacklab
import argparse


def main(output_filename):
    """Train gensim phrase model.

    Parameters
    ----------
    output_filename : {string}
        Output to the file where the phrase model is saved.
    """
    print('Training phrase model began')
    phrase_model = gensim_utils.build_gensim_phrase_model_from_sentences(blacklab.iterable_results('<s/>',
                                                                         lemma=True,
                                                                         window=0))
    phrase_model.save(output_filename)
    print('Training phrase model finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-o", "--output", type=str)
    args = parser.parse_args()

    if not args.output:
        raise Exception

    main(args.output)

    # Data/Output/phrase_model

    # python3 train_phrase_model.py -o Data/Output/phrase_model_test
