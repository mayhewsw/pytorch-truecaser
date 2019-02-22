## Truecaser

This is a simple neural truecaser written with allennlp, and based loosely on [Susanto et al, 2016](https://aclweb.org/anthology/D16-1225). They have an
implementation [here](https://gitlab.com/raymondhs/char-rnn-truecase), but being written in Lua, it's a little hard to use.

We provide a [pre-trained model](https://github.com/mayhewsw/pytorch-truecaser/releases/tag/v1.0) that can be used for truecasing English right out of the box. This model is trained on the [standard Wikipedia data split](http://www.cs.pomona.edu/~dkauchak/simplification/data.v1/data.v1.split.tar.gz) from (Coster and Kauchak, 2011), and achieves an F1 score of **93.43** on test. This is comparable to the best F1 of (Susanto et al 2016) of **93.19**.

### Requirements

* python (3.6)
* [allennlp](https://github.com/allenai/allennlp/) (0.8.1)

### Model
This model treats each sentence as a sequence of characters (spaces are included in the sequence). Each character takes a binary label
of "U" if uppercase and "L" if lowercase. For example, the word `tRuEcasIng` would take the labels `LULULLLULL`

We encode the sequence using a bidirectional LSTM with 2 hidden layers, 50 dimensional character embeddings (input), 150 dimensional hidden size, and
dropout of 0.25.


### Usage

If you just want to predict, you can run:
```bash
$ allennlp predict wiki-truecaser-model.tar.gz data/test.txt --output-file test-out.txt --include-package mylib --use-dataset-reader --silent
```

To use it programmatically (in python),

```python
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("https://github.com/mayhewsw/pytorch-truecaser/releases/download/v1.0/wiki-truecaser-model.tar.gz")
predictor.predict(sentence="Jared Smith lives in Paris .")
```


#### Training
The dataset reader requires text that has one sentence per line. The model expects tokenized text. If your text is already tokenized
(the Wiki data is), then you can use `just_spaces` as the `word_splitter` in the config. If you want to tokenize text first,
you can use `spacy`.

To train, set the values of train, validation, and test in `truecaser.config` and run:
```bash
$ allennlp train truecaser.json --include-package mylib -s /path/to/save/model/
```
