import numpy as np
import torch

from configs.configuration import config
from src import data_helper
from src.data_helper import load_label_map
from src import utils


def convert_to_tensor(word2vec_model, mini_batch):
	shortest_path_padded = [[utils.convert_and_pad(word2vec_model, d['shortest-path'], config.DEPENDENCY_TREE_LEN)] for
							d in mini_batch]
	shortest_path_padded = np.asarray(shortest_path_padded)
#	print(shortest_path_padded.shape)

	sentence_padded = [[utils.convert_and_pad(word2vec_model, d['tagged-refined-text'], config.TEXT_LEN)] for d in mini_batch]
	sentence_padded = np.asarray(sentence_padded)
#	print(sentence_padded.shape)

	x_batch = np.concatenate([shortest_path_padded, sentence_padded], axis=2)

	x_batch = np.asarray(x_batch)

#	print("x_batch shape: ", x_batch.shape)
	x_batch = torch.from_numpy(x_batch).float()
	return x_batch


def padded(original_arr, final_len):
	""" Center padding a [x*WORD_DIM] array to shape[SEQ_LEN*WORD_DIM] with zeros
	Args:
		original_arr: numpy array with shape [1, x*WORD_DIM], x MUST be less than or equal to SEQ_LEN.
		final_len: an integer denoteing the final len after padding

	Returns:
		arr: padded array
	"""
	arr = original_arr.copy()
	if arr.shape[0] == final_len * config.WORD_DIM:
		return arr

	assert arr.shape[0] < final_len * config.WORD_DIM

	half = (final_len*config.WORD_DIM - arr.shape[0]) // 2

	p = np.zeros(half)

	arr = np.append(p, arr)
	arr = np.append(arr, p)

	if arr.shape[0] != final_len * config.WORD_DIM:
		arr = np.append(arr, 0)
			
	return arr


# all_types = dict()
def convert_and_pad(word2vec, sentence, final_len):
	""" Convert shortest path with POS tags into a np matrix.
	Args:
		word2vec: a dict, word embedding dictionary.
		sentence: a list of tuple (word, POS)
			words MUST consist of latin characters ('a'-'z', 'A-Z') and digits (0-9) only.
		final_len: an integer denoteing the final len after padding
	Returns:
		m: numpy array.
	"""
	pos_map = load_label_map("configs/pos_map.txt")

	m = np.array([])

	#print("path ", path)

	for w, pos in sentence:
		# all_types[pos] = 1

		if str(w).lower() not in word2vec:
			# print("{} not in word2vec".format(w))
			continue
		word_emb = word2vec[str(w).lower()]
		# print(word_emb.shape)

		# one-hot coding
		pos_emb = np.zeros(config.pos_types)
		pos_emb[pos_map[pos]] = 1.
		#end of one hot coding

		# concatenate word embedding with POS embedding
		embedding = np.concatenate([word_emb, pos_emb])
		m = np.append(m, embedding)

	m = padded(m, final_len)
	return m


if __name__ == "__main__":
	word2vec_model = data_helper.load_word2vec()

	test_data = data_helper.load_training_data(config.TEST_PATH)
	test_data += data_helper.load_training_data(config.TRAIN_PATH)

	crossed = dict()

	for t in test_data:
		convert_and_pad(word2vec_model, t['tagged-refined-text'], config.TEXT_LEN)

	# print(*all_types)