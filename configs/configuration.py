class Config():
    def __init__(self):
        self.num_class = 19

        self.CUDA = False

        self.DEBUG = True

        self.SEQ_LEN = 16
        self.WORD_DIM = 300

        self.NUM_FILTERS = 256
        self.FILTER_SIZES = [2, 3, 4, 5]

        self.NUM_EPOCH = 10000
        self.LEARNING_RATE = 1.

        self.BATCH_SIZE = 32
        # if you want to train the whole batch, set this to very large number, say 10000000000

        self.FINE_TUNE = False
        self.CHECKPOINT_PATH = "checkpoints/checkpoint.pth"

        self.TEST_PATH = "data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"
        self.TRAIN_PATH = "data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"


config = Config()
