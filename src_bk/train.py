import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from src import data_helper
from src.model import Net
from src.test import compute_acc
from src import dependency_tree
from configs.configuration import config


def train(original_training_data, original_validate_data, net, save_name):
    optimizer = optim.Adadelta(net.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    training_data = original_training_data.copy()
    val_data = original_validate_data.copy()
    print("Training on {} datas, validating on {} datas".format(len(training_data), len(val_data)))

    if config.BATCH_SIZE > len(training_data):
        print("WARNING: batch size is larger then the length of training data,\n\
            it will be reassigned to length of training data")
        config.BATCH_SIZE = len(training_data) 

    if config.CUDA:
        net.cuda()

    final_f1_on_val = 0

    for epoch in range(config.NUM_EPOCH):
        if not net.training:
            net.train()

        training_data = original_training_data.copy()
        random.shuffle(training_data) # shuffle the training data for circular batch
        
        if len(training_data)%config.BATCH_SIZE:
            # if we cannot divide the array into chunks of batch size
            # then we will add some first elements of the array to the end
            training_data += training_data[:(config.BATCH_SIZE-len(training_data)%config.BATCH_SIZE)]

        avg_loss = 0
        for i in range(0, len(training_data), config.BATCH_SIZE):
            mini_batch = training_data[i:i+config.BATCH_SIZE]
            optimizer.zero_grad()

            target = torch.LongTensor([int(d['label-id']) for d in mini_batch])

            # print("x_batch shape: ", x_batch.shape)
            # x_batch = utils.convert_to_tensor(word2vec_model, offset1_dict, offset2_dict, mini_batch)
            x_batch_raw = [single['shortest-path'] for single in mini_batch]

            x_batch = net.convert_to_batch(x_batch_raw)

            if config.CUDA:
                x_batch = x_batch.cuda()
                target = target.cuda()

            output = net(x_batch)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()    # Does the update

            avg_loss += loss.item()

        assert len(training_data) % config.BATCH_SIZE == 0
        avg_loss /= len(training_data) / config.BATCH_SIZE
        print("Epoch {}, loss {}".format(epoch+1, avg_loss))
        if epoch % 5 == 0 or epoch == config.NUM_EPOCH - 1:
            print("Train acc {:.3f}".format(compute_acc(net, original_training_data)))
            if val_data:
                final_f1_on_val = compute_acc(net, original_validate_data)
                print("Validate acc {:.3f}".format(final_f1_on_val))
            torch.save({
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, save_name)
    return final_f1_on_val


def main():
    training_data = data_helper.load_training_data(config.TRAIN_PATH)
    # random.seed(31)
    random.shuffle(training_data)

    if not config.NO_VAL_SET: # if not training the whole dataset, then do K-fold CV
        thresh = int(0.2 * len(training_data))
        cv_score = 0
        for i in range(5):
            print("Fold {}/{}".format(i+1, 5))
            left = i * thresh
            right = (i+1) * thresh
            print("Left: {}, Right: {}".format(left, right))

            cv_score += train(training_data[:left]+training_data[right:],
                              training_data[left:right],
                              Net(),
                              config.CHECKPOINT_PATH + ".fold{}".format(i+1))
        cv_score /= 5
        print("Final CV score: {}".format(cv_score))
    else:
        val_data = []
        train(training_data, val_data, Net(), config.CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
