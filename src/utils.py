import numpy as np

from configs.configuration import config
from src import data_helper


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


def convert_and_pad(word2vec, s):
	""" Convert a sentence to a np matrix
	Args:
		word2vec : dictionary, word embedding dictionary.
		s: string, MUST consist of latin characters ('a'-'z', 'A-Z') and digits (0-9) only.
	Returns:
		m: numpy array.
	"""
	m = np.array([])
	
	s = s.split(' ')
	for w in s:
		if w in word2vec:
			m = np.append(m, word2vec[w])
			# print(word2vec[w])
		else:
			random_embedding = np.random.rand(config.WORD_DIM)
			random_embedding = (random_embedding - 0.5) * 2
			print("Added to word2vec ", w)
			word2vec[w] = random_embedding
	# print(m.shape)
	m = padded(m)
	# print(m.shape)
	return m


def test():
	word2vec_model = data_helper.load_word2vec()	
	ret = convert_and_pad(word2vec_model, "I am of a king")
	print(ret)

if __name__ == "__main__": 
	test()
