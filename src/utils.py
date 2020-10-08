import numpy as np

from configs.configuration import config
from src import data_helper
from src.data_helper import load_label_map


def padded(original_arr):
	""" Center padding a [x*WORD_DIM] array to shape[SEQ_LEN*WORD_DIM] with zeros
	Args:
		original_arr : numpy array with shape [1, x*WORD_DIM], x MUST be less than or equal to SEQ_LEN.
	Returns:
		arr: padded array
	"""
	arr = original_arr.copy()
	if arr.shape[0] == config.SEQ_LEN * config.WORD_DIM:
		return arr

	assert arr.shape[0] < config.SEQ_LEN * config.WORD_DIM

	half = (config.SEQ_LEN*config.WORD_DIM - arr.shape[0]) // 2

	p = np.zeros(half)

	arr = np.append(p, arr)
	arr = np.append(arr, p)

	if arr.shape[0] != config.SEQ_LEN * config.WORD_DIM:
		arr = np.append(arr, 0)
			
	return arr


def convert_and_pad(word2vec, path):
	""" Convert shortest path with POS tags into a np matrix.
	Args:
		word2vec: a dict, word embedding dictionary.
		path: a list of tuple (word, POS)
			words MUST consist of latin characters ('a'-'z', 'A-Z') and digits (0-9) only.
	Returns:
		m: numpy array.
	"""
	pos_map = load_label_map("configs/pos_map.txt")

	m = np.array([])
	#print("path ", path)
	for w, pos in path:
		if w not in word2vec:
			continue

		word_emb = word2vec[w]
		# print(word_emb.shape)

		# one-hot coding
		pos_emb = np.zeros(config.pos_types)
		# print("pos, posmap[pos]", pos, pos_map[pos])
		pos_emb[pos_map[pos]] = 1.
		# print(pos_emb)
		# print(pos_emb.shape)
		#end of one hot coding

		# concatenate word embedding with POS embedding
		embedding = np.concatenate([word_emb, pos_emb])
		# print(embedding.shape)
		m = np.append(m, embedding)

	# print(m.shape)
	m = padded(m)
	# print(m)
	# print(m.shape)
	return m


if __name__ == "__main__":
	word2vec_model = data_helper.load_word2vec()

	test_data = data_helper.load_training_data(config.TEST_PATH)
	# test_data += data_helper.load_training_data(config.TRAIN_PATH)

	crossed = dict()

	for t in test_data:
		convert_and_pad(word2vec_model, t['shortest-path'])
