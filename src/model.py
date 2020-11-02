import random

import torch
import torch.nn.functional as F
from torch import nn, optim

import numpy as np

from configs.configuration import config
from src import data_helper


class Net(nn.Module):

    @staticmethod
    def dict_to_emb(in_dict):
        key2index = dict()
        index = 0
        emb = []
        for key in in_dict:
            key2index[key] = index
            emb.append(in_dict[key])
            index += 1

        emb = torch.FloatTensor(emb)
        # print("word embedding shape: ", emb.shape)
        emb = nn.Embedding.from_pretrained(embeddings=emb, freeze=False)
        return emb, key2index

    @staticmethod
    def conf_to_emb(in_dict):
        emb = []
        for key in in_dict:
            if type(key) == int: # need this to filter two-way dict
                onehot = np.zeros(len(in_dict), dtype=float)
                onehot[key] = 1.
                emb.append(onehot)
        emb = torch.FloatTensor(emb)
        emb = nn.Embedding.from_pretrained(embeddings=emb, freeze=False)
        return emb

    def padded(self, original_arr, final_len):
        """ Center padding a [x*WORD_DIM] array to shape[SEQ_LEN*WORD_DIM] with zeros
        Args:
            original_arr: numpy array with shape [1, x*WORD_DIM], x MUST be less than or equal to SEQ_LEN.
            final_len: an integer denoteing the final len after padding

        Returns:
            arr: padded array
        """
        arr = original_arr.view(original_arr.shape[1])

        if arr.shape[0] == final_len * config.WORD_DIM:
            return arr

        assert arr.shape[0] < final_len * config.WORD_DIM

        half = (final_len * config.WORD_DIM - arr.shape[0]) // 2

        p = torch.FloatTensor(np.zeros(half))
        out_arr = torch.cat([p, arr, p], dim=0)

        # print("p shape", p.shape)
        # print("out_arr shape", out_arr.shape)
        # print(out_arr.shape[0], final_len * config.WORD_DIM)

        if out_arr.shape[0] != final_len * config.WORD_DIM:
            torch.cat([out_arr, torch.FloatTensor(np.zeros(config.WORD_DIM))], dim=1)

        assert(out_arr.shape[0] == final_len * config.WORD_DIM)
        out_arr = out_arr.view(1, out_arr.shape[0])
        return out_arr

    def convert_and_pad_single(self, sentence):
        m = []

        for w, pos, dep, offset1, offset2 in sentence:
            if str(w).lower() not in self.word2vec_index:
                print("{} not in word2vec".format(w))
                continue
            word_2_index = self.word2vec_index[str(w).lower()]
            # print(self.word2vec_emb)
            word_2_index = torch.LongTensor([word_2_index])
            # print("word 2 index: ", word_2_index)
            word_emb = self.word2vec_emb(word_2_index)

            pos_emb = self.POS_emb(torch.LongTensor([self.POS_dict[pos]]))

            dep_emb = self.dep_emb(torch.LongTensor([self.dep_dict[dep]]))

            offset1_emb = self.e1_offset_emb(torch.LongTensor([self.offset_index(offset1 % config.POSITION_DIM)]))
            offset2_emb = self.e2_offset_emb(torch.LongTensor([self.offset_index(offset2 % config.POSITION_DIM)]))

            # concatenate word embedding with POS embedding
            embedding = torch.cat([word_emb, pos_emb, dep_emb, offset1_emb, offset2_emb], dim=1)
            # print(embedding.shape)

            m.append(embedding)
        m = torch.cat(m, dim=1)
        # print(sentence)
        m = self.padded(m, config.SEQ_LEN)
        return m

    def convert_to_batch(self, batch_of_sentence):
        m = []
        for sentence in batch_of_sentence:
            m.append(self.convert_and_pad_single(sentence))
        out = torch.cat(m, dim=0)
        out = out.view(config.BATCH_SIZE, 1, -1)
        return out

    def __init__(self):
        super(Net, self).__init__()

        # initialize word2vec embedding
        word2vec_dict = data_helper.load_word2vec()
        self.word2vec_emb, self.word2vec_index = self.dict_to_emb(word2vec_dict)

        # initialize e1_offset_dict
        assert config.POSITION_DIM % 2 == 0
        self.offset_index = lambda x: x
        self.e1_offset_emb = nn.Embedding.from_pretrained(embeddings=torch.FloatTensor(np.identity(config.POSITION_DIM)), freeze=False)

        # initialize e2_offset_dict
        self.e2_offset_emb = nn.Embedding.from_pretrained(embeddings=torch.FloatTensor(np.identity(config.POSITION_DIM)), freeze=False)

        # initialize relation dependency dict
        self.dep_dict = data_helper.load_label_map('configs/dep_map.txt')
        self.dep_emb = self.conf_to_emb(self.dep_dict)

        # intialize part-of-speech dict
        self.POS_dict = data_helper.load_label_map('configs/pos_map.txt')
        self.POS_emb = self.conf_to_emb(self.POS_dict)

        for filter_size in config.FILTER_SIZES:
            conv = nn.Conv1d(1,\
                config.NUM_FILTERS, \
                filter_size*config.WORD_DIM, \
                stride=config.WORD_DIM)
            # in_channel,out_channel,window_size,stride
            setattr(self, 'conv_' + str(filter_size), conv)
        
        self.fc = nn.Linear(len(config.FILTER_SIZES)*config.NUM_FILTERS, config.num_class)
    
    def forward(self, raw):
        x = self.convert_to_batch(raw)
        if config.CUDA:
            x = x.cuda()
        # print("x shape", x.shape)
        z = []
        # after conv with relu activation [n][1][85*300] -> [n][20][.]
        # print(x.shape)

        for filter_size in config.FILTER_SIZES:
            t = F.relu(getattr(self, 'conv_' + str(filter_size))(x)) 
            t = F.max_pool1d(t, config.SEQ_LEN-filter_size+1)
            z.append(t)
            # after max-pool-over-time [n][20][.] -> [n][20][1]

        out = torch.cat(z, dim=1)
        # print(out.shape)

        out = out.view(config.BATCH_SIZE, config.NUM_FILTERS * len(config.FILTER_SIZES))
        # [n][20*len(FILTER_SIZES)][1] -> [n][20*len(FILTER_SIZES)], needed for nn.Linear
        # print(out.shape)

        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc(out)
        # after fc [n][20*len(FILTER_SIZES)] -> [n][config.num_class]
        # print("output shape: ", out.shape)
        return out


if __name__ == "__main__":
    net = Net()
    training_data = data_helper.load_training_data(config.DEV_PATH)
    print(training_data)
    raw_batch = [s['shortest-path'] for s in training_data[:config.BATCH_SIZE]]
    net(raw_batch)
    '''
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(count_parameters(net))

    print(net)
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    print(input.shape, target.shape)
    pass
    '''