"""Method to transform a CSV file to a fragment collection."""
import constants
import json
import random
from utils import text

""""
Creates a json object to be used as input for fragments collection
from a csv file with the following fields:
mid_leaf, media_index, media_offset, testimony_id, end_sentence_index,
label, original_sentences, start_sentence_index,main_leaf

If a leaf is prototypical, its mid_leaf left empty
"""

input_file_path = constants.OUTPUT_FOLDER + 'testimonial_fragments_updated.csv'
output_file_path = constants.OUTPUT_FOLDER + \
    'testimonial_fragments_updated.json'


def create_fragments_collection(dictionary_from_csv):
    """Create a json representation of a fragment collection."""
    all_fragments = []
    # get the main leaves

    main_leaves = list(
        set([element['main_leaf'] for element in dictionary_from_csv]))
    for main_leaf in main_leaves:
        parent_node = create_parent_node(main_leaf)
        # add prototypical instances to
        prototypical_leaves = [element for element in dictionary_from_csv
                               if ((element['main_leaf'] == main_leaf) and
                                   (element['mid_leaf'] == "nan"))]

        for prototypical_leaf in prototypical_leaves:
            prototypical_node = get_node(prototypical_leaf['testimony_id'],
                                         prototypical_leaf)
            parent_node['tree']['children'].append(prototypical_node)

        # find the mid_leaves
        mid_leaves = set([element['mid_leaf'] for element in
                         dictionary_from_csv
                         if ((element['main_leaf'] == main_leaf) and
                             (element['mid_leaf'] != "nan"))])

        for mid_leaf in mid_leaves:
            # create a parent node
            node = {}
            node['label'] = mid_leaf
            mid_node = get_node(random.randint(1, 20),
                                node, True)
            fragments = [element for element in dictionary_from_csv
                         if ((element['main_leaf'] == main_leaf) and
                             element['mid_leaf'] == mid_leaf)]
            for fragment in fragments:
                fragment_node = get_node(fragment['testimony_id'], fragment)
                mid_node['children'].append(fragment_node)

            parent_node['tree']['children'].append(mid_node)
            # get the child of this node
        all_fragments.append(parent_node)
    return all_fragments


def create_parent_node(label):
    """Generate a root node for a tree structure."""
    testimony_id = random.randint(1, 20)
    node = {}
    node['label'] = label
    fragment = {'label': label,
                'essay_id': random.randint(1, 20),
                'tree': get_node(testimony_id, node, is_parent=True)}
    fragment['tree']['label'] = label

    return fragment


def get_node(testimony_id, node, is_parent=False):
    """Generate a parent or leaf node for a tree structure."""
    if is_parent:
        return {
            'label': node['label'],
            'testimony_id': random.randint(1, 20),
            'media_index': random.randint(1, 20),
            'media_offset': random.randint(1, 20),
            'start_sentence_index': random.randint(1, 20),
            'end_sentence_index': random.randint(1, 20),
            'children': [], }
    else:
        return {'label': node['label'],
                'testimony_id': node['testimony_id'],
                'media_index': float(node['media_index']),
                'media_offset': float(node['media_offset']),
                'start_sentence_index': float(node['start_sentence_index']),
                'end_sentence_index': float(node['end_sentence_index']),
                'children': [], }


def main():
    dictionary_from_csv = text.ReadCSVasDict(input_file_path)
    result = create_fragments_collection(dictionary_from_csv)
    with open(output_file_path, 'wb') as outfile:
        json.dump(result, outfile)


if __name__ == '__main__':
    dictionary_from_csv = text.ReadCSVasDict(input_file_path)
    result = create_fragments_collection(dictionary_from_csv)

    text.write_json(output_file_path, result)
