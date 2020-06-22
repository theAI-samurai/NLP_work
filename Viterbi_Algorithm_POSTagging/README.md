# Using Vanilla Viterbi Algorithm for POS Tagging

This python code implements the Viterbi Algorithm for POS Tagging on NLTK PennTreebank Data.
The data is divided into 2 parts  : Training and Validation in the ratio 95:0.05 .

## Improvements on Vanilla Viterbi

There are 2 methods that have been used in the implementation to improve the Vanilla Viterbi Algo.
It is used to improve the accuracy on Unknown words in the Validation set.

1. **RULE Based Method :** In this We used some rules like tagging the numbers with ** "NUM" **.
and rest of all unknown words with the Most occured tag in unknown data ie  ** NOUN **.

2. **Probabilistic Based :** In this We used a prbabilistic Rule that if the word belons to the unknown category
its state probability will be equal to its Transiition Probability.
