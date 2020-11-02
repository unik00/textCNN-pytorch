import numpy as np
import torch

from configs.configuration import config
from src import data_helper
from src.data_helper import load_label_map
from src import utils


def convert_to_tensor(word2vec_model, offset1_dict, offset2_dict, mini_batch):
	shortest_path_padded = [[utils.convert_and_pad(word2vec_model,
												   	offset1_dict,
												   	offset2_dict,
												   	d['shortest-path'],
												   	config.DEPENDENCY_TREE_LEN)]
							for d in mini_batch]
	shortest_path_padded = np.asarray(shortest_path_padded)
	# print(shortest_path_padded.shape)

	'''
	sentence_padded = [[utils.convert_and_pad(word2vec_model, d['tagged-refined-text'], config.TEXT_LEN)] for d in mini_batch]
	sentence_padded = np.asarray(sentence_padded)
	# print(sentence_padded.shape)

	x_batch = np.concatenate([shortest_path_padded, sentence_padded], axis=2)
	'''
	x_batch = shortest_path_padded
	x_batch = np.asarray(x_batch)
	# print("x_batch shape: ", x_batch.shape)
	x_batch = torch.from_numpy(x_batch).float()
	return x_batch




if __name__ == "__main__":
	word2vec_model = data_helper.load_word2vec()

	test_data = data_helper.load_training_data(config.TEST_PATH)
	test_data += data_helper.load_training_data(config.TRAIN_PATH)

	crossed = dict()

	for t in test_data:
		convert_and_pad(word2vec_model, t['tagged-refined-text'], config.TEXT_LEN)

	# print(*all_types)