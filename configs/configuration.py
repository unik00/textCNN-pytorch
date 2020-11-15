import torch

class Config():
    def __init__(self):
        self.num_class = 19
        self.pos_types = 18
        self.dep_types = 44

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.CUDA = torch.cuda.is_available()

        if self.CUDA:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            print("Using GPU.")

        self.DEBUG = True

        self.DEPENDENCY_TREE_LEN = 16
        self.TEXT_LEN = 97

        self.SEQ_LEN = self.DEPENDENCY_TREE_LEN
        
        self.MAX_ABS_OFFSET = 96
        self.POSITION_DIM = 40
        self.DEP_DIM = self.dep_types
        self.POS_DIM = self.pos_types
        self.WORD_DIM = 2*(300+self.POSITION_DIM*2+self.POS_DIM) + self.DEP_DIM + 2 # 2 is for edge direction

        self.NUM_FILTERS = 64
        self.FILTER_SIZES = [1,2,3]

        self.NUM_EPOCH = 36
        self.LEARNING_RATE = 1.

        self.BATCH_SIZE = 50
        # if you want to train the whole batch, set this to very large number, say 10000000000

        self.FINE_TUNE = False
        self.CHECKPOINT_PATH = "checkpoints/checkpoint.pth"

        self.TEST_PATH = "data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"
        self.TRAIN_PATH = "data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"
        self.DEV_PATH = "data/SemEval2010_task8_all_data/SemEval2010_task8_training/DEV_FILE.TXT"

        self.NO_VAL_SET = True

config = Config()
