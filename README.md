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

- Finetune, dense position embedding, default initialization, position_dim=40
cv: 
test:

```
