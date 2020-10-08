import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from src import data_helper, utils
from src.model import Net
from src.test import compute_acc
from src import dependency_tree
from configs.configuration import config


def train(original_training_data, original_validate_data, net):
    # TODO: finish this
    optimizer = optim.Adadelta(net.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    if config.FINE_TUNE:
        checkpoint = torch.load(config.CHECKPOINT_PATH)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        net.train()
        print("Finished loading checkpoint from {}".format(config.CHECKPOINT_PATH))

    training_data = original_training_data.copy()
    val_data = original_validate_data.copy()
    print("Training on {} datas, validating on {} datas".format(len(training_data), len(val_data)))

    word2vec_model = data_helper.load_word2vec()    
    if config.BATCH_SIZE > len(training_data):
        print("WARNING: batch size is larger then the length of training data,\n\
            it will be reassigned to length of training data")
        config.BATCH_SIZE = len(training_data) 

    if config.CUDA:
        net.cuda()

    for epoch in range(config.NUM_EPOCH):
        if not net.training:
            net.train()

        training_data = original_training_data.copy()
        random.shuffle(training_data) # shuffle the training data for circular batch
        
        if len(training_data)%config.BATCH_SIZE:
            # if we cannot divide the array into chunks of batch size
            # then we will add some first elements of the array to the end
            training_data += training_data[:(config.BATCH_SIZE-len(training_data)%config.BATCH_SIZE)]
        
        for i in range(0, len(training_data), config.BATCH_SIZE):
            mini_batch = training_data[i:i+config.BATCH_SIZE]
            optimizer.zero_grad()

            target = torch.LongTensor([int(d['label-id']) for d in mini_batch])

            x_batch = [[utils.convert_and_pad(word2vec_model, d['shortest-path'])] for d in mini_batch]
            x_batch = np.asarray(x_batch)
            # print("x_batch shape: ", x_batch.shape)
            x_batch = torch.from_numpy(x_batch).float()

            if config.CUDA:
                x_batch = x_batch.cuda()
                target = target.cuda()

            # print(x_batch.type())
            output = net(x_batch)
            # print("output: ", output)
            # print("target_onehot: ", target_onehot)
            # print("shapes", output.shape, target_onehot.shape)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()    # Does the update
        
        print("Epoch {}, loss {}".format(epoch,loss.item()))
        if epoch % 5 == 0:
            print("Train acc {:.3f}".format(compute_acc(word2vec_model, net, training_data)*100))
            print("Validate acc {:.3f}".format(compute_acc(word2vec_model, net, val_data)*100))
            torch.save({
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, config.CHECKPOINT_PATH)


def main():
    training_data = data_helper.load_training_data()
    random.seed(3948)
    random.shuffle(training_data)

    # training_data = training_data[:800] # for debugging

    thresh = int(0.8 * len(training_data))
    training_data, val_data = training_data[:thresh],training_data[thresh:]
    
    net = Net()
    
    train(training_data, val_data, net)

    return


if __name__ == "__main__":
    main()
