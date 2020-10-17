import subprocess
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

    temporary_file = open("data/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/temporary_file.txt", "w")
    # key_file = open("data/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/TEST_FILE_KEY.TXT", "r")

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
                # print(i + j, original_len)
                # i + j must be less than original len to avoid duplicate
                continue
            if pred == datas[i + j]['label-id']:
                correct_cnt += 1
            else:
                # print(pred, mini_batch[j]['original-text'], mini_batch[j]['shortest-path'], datas[i + j]['label-id'])
                pass
            # if mini_batch[j]['shortest-path']:
            temporary_file.write(mini_batch[j]['num'].strip(' ') + '\t' + label_map[pred].strip(' ')+'\n')

            y_pred.append(pred)
            y_gt.append(datas[i + j]['label-id'])

    temporary_file.close()
    scorer = 'data/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/semeval2010_task8_scorer-v1.2.pl'
    temporary_file_path = 'data/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/temporary_file.txt'
    test_file_key_path = 'data/SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/TEST_FILE_KEY.TXT'
    official_result_lines = subprocess.check_output(["perl", scorer, temporary_file_path, test_file_key_path])
    official_result_lines = official_result_lines.decode('utf-8').split('\n')
    official_score = float(official_result_lines[-2][-10:-5])
    # print(official_score)
    # print(official_result_lines)
    acc = 1.0*correct_cnt / original_len * 100
    print("Accuracy: {:.2f}, F1-score: {:.2f}".format(acc, official_score))
    return official_score


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

