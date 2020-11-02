import pickle

from gensim.models import KeyedVectors

from src import data_helper
from configs.configuration import config
import numpy as np


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, 3)


if __name__ == "__main__":
    word2vec_model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

    test_data = data_helper.load_training_data(config.TEST_PATH)
    test_data += data_helper.load_training_data(config.TRAIN_PATH)

    crossed = dict()

    for t in test_data:
        words = [w[0].lower() for w in t['shortest-path']]

        for word in words:

            if word in word2vec_model:
                crossed[word] = word2vec_model[word]
            else:
                print("not found {}".format(word))
                crossed[word] = np.random.randn(300)
                print(crossed[word])

    save_obj(crossed, "word2vec_crossed")
