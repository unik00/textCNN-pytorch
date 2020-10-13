## Text-CNN implementation using Tensorflow 2 (mostly tf.keras)  
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
    
##### Done 
```
<<< (9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL >>>:
Confusion matrix:
         C-E  C-W  C-C  E-D  E-O  I-A  M-C  M-T  P-P  _O_ <-- classified as
      +--------------------------------------------------+ -SUM- xDIRx skip  ACTUAL
  C-E | 296    0    0    0    9    1    0    1    2   15 |  324     4     0    328
  C-W |   2  209    5    3    1    9   13    5    2   32 |  281    31     0    312
  C-C |   0    4  160   10    2    0    0    1    0   11 |  188     4     0    192
  E-D |   0    2   10  260    0    0    0    2    1   16 |  291     1     0    292
  E-O |   2    1    0   12  221    1    0    1    3   16 |  257     1     0    258
  I-A |   0    4    1    2    2  101    1    3    9   31 |  154     2     0    156
  M-C |   0    6    0    0    2    3  195    4    2   20 |  232     1     0    233
  M-T |   1    2    0    2    0    0    3  228    3   19 |  258     3     0    261
  P-P |   4    4    2    1    9   10    0    5  167   24 |  226     5     0    231
  _O_ |  20   25   25   25   32   20   39   39   26  203 |  454     0     0    454
      +--------------------------------------------------+
 -SUM-  325  257  203  315  278  145  251  289  215  387   2665    52     0   2717

Coverage = 2717/2717 = 100.00%
Accuracy (calculated for the above confusion matrix) = 2040/2717 = 75.08%
Accuracy (considering all skipped examples as Wrong) = 2040/2717 = 75.08%
Accuracy (considering all skipped examples as Other) = 2040/2717 = 75.08%

Results for the individual relations:
             Cause-Effect :    P =  296/( 325 +   4) =  89.97%     R =  296/ 328 =  90.24%     F1 =  90.11%
          Component-Whole :    P =  209/( 257 +  31) =  72.57%     R =  209/ 312 =  66.99%     F1 =  69.67%
        Content-Container :    P =  160/( 203 +   4) =  77.29%     R =  160/ 192 =  83.33%     F1 =  80.20%
       Entity-Destination :    P =  260/( 315 +   1) =  82.28%     R =  260/ 292 =  89.04%     F1 =  85.53%
            Entity-Origin :    P =  221/( 278 +   1) =  79.21%     R =  221/ 258 =  85.66%     F1 =  82.31%
        Instrument-Agency :    P =  101/( 145 +   2) =  68.71%     R =  101/ 156 =  64.74%     F1 =  66.67%
        Member-Collection :    P =  195/( 251 +   1) =  77.38%     R =  195/ 233 =  83.69%     F1 =  80.41%
            Message-Topic :    P =  228/( 289 +   3) =  78.08%     R =  228/ 261 =  87.36%     F1 =  82.46%
         Product-Producer :    P =  167/( 215 +   5) =  75.91%     R =  167/ 231 =  72.29%     F1 =  74.06%
                   _Other :    P =  203/( 387 +   0) =  52.45%     R =  203/ 454 =  44.71%     F1 =  48.28%

Micro-averaged result (excluding Other):
P = 1837/2330 =  78.84%     R = 1837/2263 =  81.18%     F1 =  79.99%

MACRO-averaged result (excluding Other):
P =  77.93%	R =  80.37%	F1 =  79.05%



<<< The official score is (9+1)-way evaluation with directionality taken into account: macro-averaged F1 = 79.05% >>>
```
##### Problems

##### Plan
- Evaluate with scorer
- 5-fold validation
- Convert pytorch to tf.keras
