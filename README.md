**Text-CNN implementation using Tensorflow 2 (mostly tf.keras though)** \
[Yoon Kim 2014 - https://arxiv.org/pdf/1408.5882.pdf]

Word2vec pre-trained models: \
*Google-News-all:* https://github.com/mmihaltz/word2vec-GoogleNews-vectors \
*Google-News-slim:* https://github.com/eyaler/word2vec-slim
*Google-News-slim with unavailaible keys initialized:* https://drive.google.com/file/d/1TMkEMdO_BMbdS1Bip5wyMFyo-0E1N4-l/view?usp=sharing

**Finished:**
- Train
- Data preprocessing
- Evaluation (untested)
- Gộp softmax + MSELoss thành CrossEntropyLoss
- Implement load checkpoints
- Initialized unavailable keys from word2vec with random floats

**Problems:**


**Plan:**
- Convert pytorch to tf.keras
