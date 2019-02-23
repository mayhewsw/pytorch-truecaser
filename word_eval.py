# This code taken verbatim from https://gitlab.com/raymondhs/char-rnn-truecase/blob/master/word_eval.py
# Thanks to the authors.

import sys, codecs

gold = sys.argv[1]
pred = sys.argv[2]

gold_sent = codecs.open(gold,'r',encoding='utf8').readlines()
pred_sent = codecs.open(pred,'r',encoding='utf8').readlines()

num_correct = 0
num_changed_correct = 0
num_gold = 0
num_proposed = 0
total = 0
for i in range(len(pred_sent)):
    words = gold_sent[i].strip().split()
    pred_words = pred_sent[i].strip().split()
    for k in range(len(words)):
        if pred_words[k] != pred_words[k].lower():
            num_proposed += 1
        if words[k] != words[k].lower():
            num_gold += 1
        if words[k] == pred_words[k]:
            num_correct += 1
            if words[k] != words[k].lower():
                num_changed_correct += 1
    total += len(words)
acc = num_correct * 100.0 / total
try:
    P = float(num_changed_correct)/num_proposed
    R = float(num_changed_correct)/num_gold
    F = 2*P*R/(P+R)
except:
    P = 0
    R = 0
    F = 0
print('Accuracy: %.2f' % acc)
print('Precision: %.2f' % (P*100))
print('Recall: %.2f' % (R*100))
print('F1: %.2f' % (F*100))
print('%.2f & %.2f & %.2f & %.2f' % (acc, P*100, R*100, F*100))
