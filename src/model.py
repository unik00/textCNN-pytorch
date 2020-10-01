import torch
import torch.nn.functional as F
import random
from configs.configuration import config

from torch import nn, optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # print(config.FILTER_SIZES)
        for filter_size in config.FILTER_SIZES:
            conv = nn.Conv1d(1,\
                config.NUM_FILTERS, \
                filter_size*config.WORD_DIM, \
                stride=config.WORD_DIM)
            # in_channel,out_channel,window_size,stride
            setattr(self, 'conv_' + str(filter_size), conv)
        
        self.fc = nn.Linear(len(config.FILTER_SIZES)*config.NUM_FILTERS, config.num_class)
    
    def forward(self, x):
        """ Overiding forward method of torch.nn.Module
        Parameters
        ----------
        x : torch tensor
            shape [n, 1, SEQ_LEN * WORD_DIM], where n is batchsize.
        
        Returns
        ----------
        out : torch tensor
            shape [n, 1, num_class]
        """
        
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

        out = out.view(config.BATCH_SIZE, -1, config.NUM_FILTERS * len(config.FILTER_SIZES))
        # [n][20*len(FILTER_SIZES)][1] -> [n][1][20*len(FILTER_SIZES)], needed for nn.Linear
        # print(out.shape)

        out = F.dropout(out, p=0.5, training=self.training)
        out = F.softmax(self.fc(out), dim=2) # after fc with softmax activation [n][1][20] -> [n][1][10]
        # print(out.shape)
        return out

if __name__ == "__main__":
    net = Net()
    print(net)
    pass