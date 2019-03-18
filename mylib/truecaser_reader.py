from typing import Iterator, List, Dict
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.commands.train import *
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import *
from allennlp.data.tokenizers.word_splitter import WordSplitter,SpacyWordSplitter
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance

@DatasetReader.register("truecaser_reader")
class TrueCaserDatasetReader(DatasetReader):

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 word_splitter: WordSplitter = None) -> None:
        super().__init__(lazy=False)
        self._word_splitter = word_splitter or SpacyWordSplitter()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        token_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": token_field}

        if tags:
            label_field = SequenceLabelField(labels=tags, sequence_field=token_field)
            fields["tags"] = label_field

        return Instance(fields)

    def _read(self, file_path: str) -> Iterator[Instance]:

        with open(file_path) as f:
            for line in f:
                # I want a sentence represented by a string.
                tokenized_sent = " ".join(map(str, self._word_splitter.split_words(line.strip())))
                chars = [Token(c) for c in tokenized_sent.lower()]
                case_labels = ["U" if char.isupper() else "L" for char in tokenized_sent]
                if len(chars) != len(case_labels):
                    print("Mismatching sentence lengths!", tokenized_sent)
                    continue
                yield self.text_to_instance(chars, case_labels)

if __name__ == "__main__":
    dr = TrueCaserDatasetReader()
    for i in dr._read("tmp"):
        print(i)
