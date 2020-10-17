import torch


class Config():
    def __init__(self):
        self.num_class = 19
        self.pos_types = 18

        self.CUDA = torch.cuda.is_available()
        if self.CUDA:
            print("Using GPU.")
        self.DEBUG = True

        self.DEPENDENCY_TREE_LEN = 16
        self.TEXT_LEN = 85
        # self.SEQ_LEN = self.DEPENDENCY_TREE_LEN + self.TEXT_LEN
        self.SEQ_LEN = self.DEPENDENCY_TREE_LEN

        self.WORD_DIM = 300 + self.pos_types

        self.NUM_FILTERS = 128
        self.FILTER_SIZES = range(2,11)

        self.NUM_EPOCH = 3
        self.LEARNING_RATE = 1.

        self.BATCH_SIZE = 32
        # if you want to train the whole batch, set this to very large number, say 10000000000

        self.FINE_TUNE = False
        self.CHECKPOINT_PATH = "checkpoints/checkpoint.pth"

        self.TEST_PATH = "data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"
        self.TRAIN_PATH = "data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"

        self.NO_VAL_SET = False

config = Config()
