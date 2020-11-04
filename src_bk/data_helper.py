import pickle

from configs.configuration import config
from src import dependency_tree


def load_word2vec():
    """
    Returns:
        model : a dict for mapping word embedding.
    """
    def load_obj(name):
        with open('obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    print("Loading word2vec model...")

    # use the slim version in debugging mode for quick loading
    # model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300-SLIM.bin', binary=True)
    model = load_obj("word2vec_crossed")
    print("Finished loading")

    return model


def load_offsetdict(type):
    """
    Params:
        type: a string "1" or "2"
    """
    def load_obj(name):
        with open('obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    print("Loading offset dictionary...")

    # use the slim version in debugging mode for quick loading
    # model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300-SLIM.bin', binary=True)
    name = "pos{}_emb".format(type)

    model = load_obj(name)
    print("Finished loading {}".format(name))

    return model


def find_text_in_tag(st, tag):
    """ Find the first text between given pair of tags, returns both the text and position

    Args:
        st: string, e.g. "Hello <e1>world everybody</e1>".
        tag: tag, e.g. "e1".
    Returns:
        content: string, e.g. "world everybody".
        position: an integer
    """
    if tag == "e1":
        st = st.replace("<e2>", "")
        st = st.replace("</e2>", "")
    elif tag == "e2":
        st = st.replace("<e1>", "")
        st = st.replace("</e1>", "")

    for i in range(len(st) - (len(tag)+2) + 1): # +2 is for < and >
        if st[i:i+len(tag)+2] == "<" + tag + ">":
            for j in range(i+1, len(st) - (len(tag)+3) + 1):
                if st[j:j+len(tag)+3] == "</" + tag + ">":
                    return st[i+len(tag)+2:j], i - 1

    print("ERROR: tag \"{}\" in string \"{}\" not found!".format(tag, st))


def refined_text(text):
    """ Refine the text and tagged with POS as required by utils.convert_and_pad().
    Params:
        text: original text from SemeVal
            e.g: "The <e1>child</e1> was carefully wrapped and bound into the <e2>'cradle'</e2> by means of a cord."

    Returns:
        text, e.g: The child was carefully wrapped and bound into the cradle by means of a cord
    """
    text = text.replace('<e1>','')
    text = text.replace('</e1>','')
    text = text.replace('<e2>','')
    text = text.replace('</e2>','')
    text = text[1:-1] # trim quotes
    # text = text.replace('"','')
    # text = text.replace(',','')
    # text = text.replace('.','')
    # text = text.replace(';','')
    # text = text.replace('`','')
    # text = text.replace('\'','')
    # text = text.replace('(','')
    # text = text.replace(')','')
    # text = text.replace('/','')

    return text


def load_label_map(location="configs/label_map.txt"):
    """ Load the label map
    Returns:
        ret: a dict
    """
    ret = dict()
    num_class = 0
    with open(location) as f:
        for line in f:
            line = line.strip('\n')
            index, relation = line.split(' ')
            ret[relation] = int(index)
            ret[int(index)] = relation
            num_class += 1
    return ret


def load_training_data(data_loc='data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'):
    """
    Returns:
        ret: list of dictionaries
            content of ret[.]:
            'num': index of the data
            'source': source vertex
            'dest': destination vertex
            'label-str': original label
            'label-id': label converted to integer
            'original-text': original text containing e1 and e2
            'refined-text': cleaned text (removed <, >,... ) and tagged with POS
            'shortest-path': shortest path between e1 and e2 in dependency tree
    """

    label_map = load_label_map("configs/label_map.txt")

    ret = []

    max_length = 0
    with open(data_loc, 'r') as file:
        for i, line in enumerate(file):
            if i % 4 == 0: # is sentence
                edge_dict = dict()
                edge_dict['original-text'] = line

                num, text = line.split('\t')
                text = text.strip('\n') # remove line break

                max_length = max(max_length, len(text.split(' ')))

                e1_text, e1_position = find_text_in_tag(text, "e1")
                # print("e1 position: ", e1_position)
                e2_text, e2_position = find_text_in_tag(text, "e2")
                # return position of e2

                edge_dict['source'] = e1_text
                edge_dict['dest'] = e2_text

                text = refined_text(text)
                en_nlp_doc = dependency_tree.en_nlp(text)

                # print(line)

                for sent in en_nlp_doc.sents:
                    tagged_refined_text = []
                    for w in sent:
                        tagged_refined_text.append((w, w.pos_))
                    break

                edge_dict['refined-text'] = text
                edge_dict['tagged-refined-text'] = tagged_refined_text
                edge_dict['num'] = num
            elif i % 4 == 1: # is label
                text = line.strip('\n')

                refined = text

                edge_dict['label-str'] = refined
                edge_dict['label-id'] = label_map[refined]
                edge_dict['shortest-path'] = dependency_tree.get_shortest_path(
                    en_nlp_doc,
                    edge_dict["refined-text"],
                    e1_position,
                    e2_position,
                    edge_dict['original-text']
                )
                # print(edge_dict['original-text'])
                '''
                for word, pos, dep, e1_offset, e2_offset in edge_dict['shortest-path']:
                    # print(word, e1_offset, e2_offset)
                    if e1_offset not in off1_dict:
                        off1_dict[e1_offset] = 0
                    if e2_offset not in off2_dict:
                        off2_dict[e2_offset] = 0
                    off1_dict[e1_offset] += 1
                    off2_dict[e2_offset] += 1
                '''
            elif i % 4 == 2: # is comment
                # print(edge_dict)
                # We don't train datas which we cannot parse dependency
                if data_loc == config.TRAIN_PATH and not edge_dict['shortest-path']:
                     print("Removed this data from train set because we cannot parse:\n \t{}".format(edge_dict['original-text']))
                else:
                    ret.append(edge_dict)
    # print("max_length: {}".format(max_length))
    # print(*ret)
    return ret


if __name__ == "__main__":
    import numpy as np
    training_data = load_training_data(config.TEST_PATH) + load_training_data(config.TRAIN_PATH)
    print(training_data)
    clgt = 0
    for d in training_data:
        for _, _, _, o1, o2 in d['shortest-path']:
            clgt = max(clgt, abs(o1))
            clgt = max(clgt, abs(o2))
    print(clgt)

    '''
    model = load_word2vec()


    print(off1_dict)
    print("---")
    print(off2_dict)
    for key in off1_dict:
        off1_dict[key] = np.random.randn(config.POSITION_DIM)

    for key in off2_dict:
        off2_dict[key] = np.random.randn(config.POSITION_DIM)


    def save_obj(obj, name):
        with open('obj/' + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, 3)


    print(off1_dict)
    print(off2_dict)
    save_obj(off1_dict, "pos1_emb")
    save_obj(off2_dict, "pos2_emb")

    print("Number of class: ", config.num_class)
    print("Total training data: {}".format(len(training_data)))

    '''
