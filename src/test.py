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
        acc: float
    """
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
        x_batch = [[utils.convert_and_pad(word2vec_model, d['shortest-path'])] for d in mini_batch]
        x_batch = np.asarray(x_batch)
        # print("x_batch shape: ", x_batch.shape)
        x_batch = torch.from_numpy(x_batch).float()
        if use_cuda:
            output_batch = net(x_batch.cuda()).detach().cpu().numpy()
        else:
            output_batch = net(x_batch).detach().numpy()
    
        for j, output in enumerate(output_batch):
            pred = np.argmax(output)
            
            if i + j >= original_len:
                # i + j must be less than original len to avoid duplicate
            	continue
            if (pred == datas[i + j]['label-id']):
               	correct_cnt += 1
            y_pred.append(pred)
            y_gt.append(datas[i + j]['label-id'])

    acc = 1.0*correct_cnt / original_len * 100
    f1 = f1_score(y_gt, y_pred, average='macro')
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
    test_data = data_helper.load_training_data("data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT")
    print(compute_acc(word2vec_model, model, test_data, False))

