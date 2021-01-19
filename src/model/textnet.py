import torch.nn.functional as F

from src.model.emb import *


class TextNet(nn.Module):
    def __init__(self):
        super(TextNet, self).__init__()

        # embedding layers
        self.word = WordEmb()
        self.dep = DependencyEmb()
        self.part_of_speech = POSEmb()
        self.offset1 = PositionEmb()
        self.offset2 = PositionEmb()
        self.direction = EdgeDirectionEmb()

        # convolution layers
        for filter_size in config.FILTER_SIZES:
            conv = nn.Conv1d(1, config.NUM_FILTERS, filter_size * config.WORD_DIM, stride=config.WORD_DIM)
            # in_channel,out_channel,window_size,stride
            setattr(self, 'conv_' + str(filter_size), conv)

        self.fc = nn.Linear(len(config.FILTER_SIZES) * config.NUM_FILTERS, config.num_class)

    @staticmethod
    def padded(original_arr, final_len):
        """ Center padding a [x*WORD_DIM] array to shape[SEQ_LEN*WORD_DIM] with zeros
        Args:
            original_arr: numpy array with shape [1, x*WORD_DIM], x MUST be less than or equal to SEQ_LEN.
            final_len: an integer denoteing the final len after padding

        Returns:
            arr: padded array
        """
        arr = original_arr.view(original_arr.shape[1])

        if arr.shape[0] == final_len * config.WORD_DIM:
            return arr.view(1, arr.shape[0])

        assert arr.shape[0] < final_len * config.WORD_DIM, "arr shape: {}, final_len: {}, word_dim: {}".format(
            arr.shape[0],
            final_len,
            config.WORD_DIM)

        half = (final_len * config.WORD_DIM - arr.shape[0])

        p = torch.FloatTensor(np.zeros(half)).to(config.device)
        out_arr = torch.cat([arr, p], dim=0)

        assert (out_arr.shape[0] == final_len * config.WORD_DIM)
        out_arr = out_arr.view(1, out_arr.shape[0])
        return out_arr

    def convert_and_pad_single(self, sentence):
        m = []

        last_emb = None
        last_edge = None

        # iterate the sentence
        for w, pos, dep, dep_dir, offset1, offset2 in sentence:

            if str(w).lower() not in self.word2vec_index:
                continue

            # get word embedding
            word_emb = self.word(str(w).lower())

            # get part of speech embedding
            pos_emb = self.part_of_speech(pos)

            # get offset embedding
            offset1_emb = self.offset1(offset1)
            offset2_emb = self.offset2(offset2)

            dep_emb = None
            if dep is not None:
                dep_emb = self.dep(dep)

            # concatenate word embedding with POS embedding
            embedding = torch.cat([word_emb, offset1_emb, offset2_emb, pos_emb], dim=1)

            if last_emb is not None:
                assert last_edge is not None
                new_edge_embedding = torch.cat([last_emb, last_edge, embedding], dim=1)
                m.append(new_edge_embedding)
                assert new_edge_embedding.shape[1] == config.WORD_DIM

            dep_dir_emb = self.direction(dep_dir)
            if dep is not None:  # not the last token in sentence
                last_edge = torch.cat([dep_emb, dep_dir_emb], dim=1)
                last_emb = embedding

        if len(m) > 0:
            m = torch.cat(m, dim=1)
        else:
            m = torch.FloatTensor(np.zeros(config.WORD_DIM)).to(config.device).view(1, config.WORD_DIM)
        m = self.padded(m, config.SEQ_LEN)
        return m

    def convert_to_batch(self, batch_of_sentence):
        m = []
        for sentence in batch_of_sentence:
            t = self.convert_and_pad_single(sentence)
            m.append(t)

        out = None
        try:
            out = torch.cat(m, dim=0)
        except:
            exit()
        out = out.view(config.BATCH_SIZE, 1, -1)
        return out

    def forward(self, x):
        z = []

        for filter_size in config.FILTER_SIZES:
            t = F.relu(getattr(self, 'conv_' + str(filter_size))(x))
            t = F.max_pool1d(t, config.SEQ_LEN - filter_size + 1)
            z.append(t)

        out = torch.cat(z, dim=1)

        out = out.view(config.BATCH_SIZE, config.NUM_FILTERS * len(config.FILTER_SIZES))

        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    net = TextNet()
    training_data = data_helper.load_training_data(config.DEV_PATH)
    print(training_data)
    raw_batch = [s['shortest-path'] for s in training_data[:config.BATCH_SIZE]]
    batch = net.convert_to_batch(raw_batch)
    net(batch)
