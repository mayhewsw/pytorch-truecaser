from overrides import overrides
from typing import List
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import Token


@Predictor.register('truecaser-predictor')
class TruecaserPredictor(Predictor):
    """
    This is basically a copy of the SentenceTagger from allennlp. It is
    modified to dump output in a more sensible manner.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = JustSpacesWordSplitter()
        self.model = model

    def predict(self, sent):
        js = {"sentence" : sent}
        return self.predict_instance(self._json_to_instance(js))

    @overrides
    def predict_instance(self, sent: Instance) -> JsonDict:
        output = super().predict_instance(sent)
        #output["chars"] = sent["tokens"]
        output["words"] = sent["tokens"].tokens
        return output

    @overrides
    def predict_batch_instance(self, sents: List[Instance]) -> List[JsonDict]:
        outputs = super().predict_batch_instance(sents)
        for i,sent in enumerate(sents):
            #output["chars"] = sent["tokens"]
            outputs[i]["words"] = sent["tokens"].tokens
        return outputs

    
    def load_line(self, line: str) -> JsonDict:
        """
        This will usually be overridden with use_dataset_reader = True on the command line.
        :param line:
        :return:
        """
        return {"sentence": line}

    def dump_line(self, outputs: JsonDict):
        newd = {}
        tags = outputs["tags"]
        chars = outputs["words"]

        # all chars are lower case by default.
        out = []
        for token,t in zip(chars,tags):
            c = token.text
            if t == "U":
                c = c.upper()
            out.append(c)

        return "".join(out) + "\n"

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        tokenized_sent = " ".join(map(str, self._tokenizer.split_words(sentence)))
        chars = [Token(c) for c in tokenized_sent.lower()]
        return self._dataset_reader.text_to_instance(chars)
