## Truecaser

This is a simple neural truecaser written with allennlp, and based loosely on [Susanto et al, 2016](https://aclweb.org/anthology/D16-1225). They have an
implementation [here](https://gitlab.com/raymondhs/char-rnn-truecase), but being written in Lua, it's a little hard to use. 

We provide a [pre-trained model]() that can be used for truecasing English right out of the box. This model is trained on the [standard Wikipedia data split](http://www.cs.pomona.edu/~dkauchak/simplification/data.v1/data.v1.split.tar.gz) from (Coster and Kauchak, 2011), and achieves an F1 score of ?? on test. This is comparable to the best F1 of (Susanto et al 2016) of **93.19**.

### Requirements

* python (3.6)
* [allennlp](https://github.com/allenai/allennlp/) (0.8.1)

### Usage

If you just want to predict, you can run:
```bash
$ allennlp predict wiki-truecaser-model.tar.gz test.txt -o test-out.txt --include-package mylib --use-dataset-reader --silent
```


#### Training
The dataset reader requires text that has one sentence per line. The model expects tokenized text. If your text is already tokenized
(the Wiki data is), then you can use `just_spaces` as the `word_splitter` in the config. If you want to tokenize text first,
you can use `spacy`.

To train, set the values of train, validation, and test in `truecaser.config` and run:
```bash
$ allennlp train truecaser.json --include-package mylib -s /path/to/save/model/
```




