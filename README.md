```
<<< (9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL >>>:

 Confusion matrix:
          C-E  C-W  C-C  E-D  E-O  I-A  M-C  M-T  P-P  _O_ <-- classified as
       +--------------------------------------------------+ -SUM- xDIRx skip  ACTUAL
   C-E | 300    0    0    0    4    1    0    2    2   12 |  321     3     0    324
   C-W |   1  235    5    1    1   11   11    6    3   29 |  303     8     0    311
   C-C |   0    5  160    7    2    0    1    1    1   11 |  188     4     0    192
   E-D |   0    2    7  264    0    0    0    1    1   16 |  291     1     0    292
   E-O |   3    1    1    4  226    1    0    1    3   16 |  256     1     0    257
   I-A |   0    4    1    1    1  118    0    4    5   21 |  155     0     0    155
   M-C |   0    4    0    0    2    0  206    1    0   19 |  232     0     0    232
   M-T |   1    5    0    1    2    0    2  229    1   19 |  260     1     0    261
   P-P |   5    1    2    0    4    9    0    3  177   27 |  228     2     0    230
   _O_ |  16   25   18   23   27   24   29   33   26  231 |  452     0     0    452
       +--------------------------------------------------+
  -SUM-  326  282  194  301  269  164  249  281  219  401   2686    20     0   2706

 Coverage = 2706/2706 = 100.00%
 Accuracy (calculated for the above confusion matrix) = 2146/2706 = 79.31%
 Accuracy (considering all skipped examples as Wrong) = 2146/2706 = 79.31%
 Accuracy (considering all skipped examples as Other) = 2146/2706 = 79.31%

 Results for the individual relations:
              Cause-Effect :    P =  300/( 326 +   3) =  91.19%     R =  300/ 324 =  92.59%     F1 =  91.88%
           Component-Whole :    P =  235/( 282 +   8) =  81.03%     R =  235/ 311 =  75.56%     F1 =  78.20%
         Content-Container :    P =  160/( 194 +   4) =  80.81%     R =  160/ 192 =  83.33%     F1 =  82.05%
        Entity-Destination :    P =  264/( 301 +   1) =  87.42%     R =  264/ 292 =  90.41%     F1 =  88.89%
             Entity-Origin :    P =  226/( 269 +   1) =  83.70%     R =  226/ 257 =  87.94%     F1 =  85.77%
         Instrument-Agency :    P =  118/( 164 +   0) =  71.95%     R =  118/ 155 =  76.13%     F1 =  73.98%
         Member-Collection :    P =  206/( 249 +   0) =  82.73%     R =  206/ 232 =  88.79%     F1 =  85.65%
             Message-Topic :    P =  229/( 281 +   1) =  81.21%     R =  229/ 261 =  87.74%     F1 =  84.35%
          Product-Producer :    P =  177/( 219 +   2) =  80.09%     R =  177/ 230 =  76.96%     F1 =  78.49%
                    _Other :    P =  231/( 401 +   0) =  57.61%     R =  231/ 452 =  51.11%     F1 =  54.16%

 Micro-averaged result (excluding Other):
 P = 1915/2305 =  83.08%     R = 1915/2254 =  84.96%     F1 =  84.01%

 MACRO-averaged result (excluding Other):
 P =  82.24%\tR =  84.38%\tF1 =  83.25%



 <<< The official score is (9+1)-way evaluation with directionality taken into account: macro-averaged F1 = 83.25% >>>
 
```
### problem 
overfitting after 20-th epoch

### to-do
multichannel
