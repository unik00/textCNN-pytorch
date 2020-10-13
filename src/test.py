import torch
import numpy as np
from sklearn.metrics import f1_score

from src.model import Net
from src import data_helper, utils
from configs.configuration import config


def compute_acc(word2vec_model, net, original_datas, use_cuda=config.CUDA):
    """ Compute accuracy given model and data
    Args:
        net: Net instance
        word2vec_model: dict
        original_datas: data for evaluation

    Returns:
        acc: double
    """
    label_map = data_helper.load_label_map()
    net.eval()
    datas = original_datas.copy()

    original_len = len(datas)
    if len(datas) % config.BATCH_SIZE:
        datas += datas[:(config.BATCH_SIZE-len(datas)%config.BATCH_SIZE)]
    
    correct_cnt = 0

    y_gt = []
    y_pred = []
    for i in range(0, len(datas), config.BATCH_SIZE):
        mini_batch = datas[i:i+config.BATCH_SIZE]
        x_batch = utils.convert_to_tensor(word2vec_model, mini_batch)

        if use_cuda:
            output_batch = net(x_batch.cuda()).detach().cpu().numpy()
        else:
            output_batch = net(x_batch).detach().numpy()
    
        for j, output in enumerate(output_batch):
            pred = np.argmax(output)

            # if we fail to parse dependency tree, we assume that the class if Other
            if not mini_batch[j]['shortest-path']:
                pred = 0

            if i + j >= original_len:
                print(i + j, original_len)
                # i + j must be less than original len to avoid duplicate
                continue
            if pred == datas[i + j]['label-id']:
                correct_cnt += 1
            else:
                # print(pred, mini_batch[j]['original-text'], mini_batch[j]['shortest-path'], datas[i + j]['label-id'])
                pass
            if mini_batch[j]['shortest-path']:
                print(mini_batch[j]['num'].strip(' '), '\t', label_map[pred].strip(' '))
            # print(mini_batch[j]['num'].strip(' '), '\t', mini_batch[j]['label-str'].strip(' '))

            y_pred.append(pred)
            y_gt.append(datas[i + j]['label-id'])

    acc = 1.0*correct_cnt / original_len * 100
    f1 = f1_score(y_gt, y_pred, average=None)
    print("Acc: {} %, F1-score: {}".format(acc, f1))
    return f1


if __name__ == "__main__":
    model = Net()
    checkpoint = torch.load("checkpoints/checkpoint.pth", map_location=torch.device('cpu') )
    model.load_state_dict(checkpoint['net_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(epoch, loss)

    word2vec_model = data_helper.load_word2vec()

    test_data = data_helper.load_training_data(config.TEST_PATH)
    print("len test data", len(test_data))
    # test_data = data_helper.load_training_data(config.TRAIN_PATH)

    print(compute_acc(word2vec_model, model, test_data, False))

