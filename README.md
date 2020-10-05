**Text-CNN implementation using Tensorflow 2 (mostly tf.keras though)** \
[Yoon Kim 2014 - https://arxiv.org/pdf/1408.5882.pdf]

Word2vec pre-trained models: \
*Google-News-all:* https://github.com/mmihaltz/word2vec-GoogleNews-vectors \
*Google-News-slim:* https://github.com/eyaler/word2vec-slim

**Finished:**
- Train
- Data preprocessing
- Evaluation (untested)
- Gộp softmax + MSELoss thành CrossEntropyLoss


**Problems:**
- Model word2vec của Google lớn nên dự kiến sẽ cross các từ mình cần dùng trên Google Collab rồi tải về, hiện giờ vẫn dùng word2vec SLIM


**Plan:**
- Convert pytorch to tf.keras
- Implement load checkpoints
