class Config():
	def __init__(self):
		self.num_class = 10
		
		self.CUDA = False 

		self.DEBUG = True

		self.SEQ_LEN = 85
		self.WORD_DIM = 300

		self.NUM_FILTERS = 20
		self.FILTER_SIZES = list(range(3, 11))
		
		self.NUM_EPOCH = 10000
		self.LEARNING_RATE = 1.
		
		self.BATCH_SIZE = 1
		# if you want to train the whole batch, set this to very large number, say 10000000000


config = Config()
