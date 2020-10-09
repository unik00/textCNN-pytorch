##Text-CNN implementation using Tensorflow 2 (mostly tf.keras)  
For semantic relation extraction

<h5>Word2vec pre-trained models</h5>
*Google-News-all:* https://github.com/mmihaltz/word2vec-GoogleNews-vectors  
*Google-News-slim:* https://github.com/eyaler/word2vec-slim  
*Google-News-slim with unavailaible keys initialized:* https://drive.google.com/file/d/1XZ8OizYBCh3nJn9uz7vbYpuivOqcuEfM/view?usp=sharing 


**Requirements** \
*If you are financially poor like I am, and forced to use Google Collab,
please remember to check if the versions match.
Otherwise pretrained model will suffer from 1-2% accuracy degradation.*

    nltk==3.2.5
    spacy==2.2.4
    numpy==1.18.5
    torch==1.6.0
    
#####Done 
After fixing version mismatch and adding POS
- micro-F1: 74.236 -> 75.295
- macro-F1: 70.589 -> 71.447

#####Problems

#####Plan
- 5-fold validation
- Convert float tensor to double tensor
- Convert pytorch to tf.keras
