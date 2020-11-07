```
- Static, no position embedding:
cv: 79.44
test: 81.30

- Finetune, no position embedding, skip unvailable words in word2vec:
cv: 79.87
test: 81.73

- Finetune, with position embedding modulo position_dim=20:
cv: 79.75
test: 

- Finetune, 
with position embedding modulo position_dim=10, 
filter_size=range(2,6):
cv: 79.29
test:

- Finetune, 
with position embedding modulo position_dim=10, 
filter_size=range(2,16),
num_epoch per fold = 25:
cv: 79.58
test: 81.12

- Finetune, 
initialized word2vec, 
filtersizes=[2,3,4,5], 
numfilters=128,
only shortest path + position:
cv: 81.31
test: 82.2

- Finetune, 
initialized word2vec, 
filtersizes=[2 to 15], 
numfilters=128,
only shortest path + position:
cv: 
test: 81.8

```
