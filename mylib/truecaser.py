from typing import Dict, Optional, List, Any

from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import F1Measure


@Model.register("truecaser")
class TrueCaser(SimpleTagger):
    """
    This is a simple child of SimpleTagger. The only difference is that
    I wanted to include an F1 measure in the metrics (even though this is
    character F1 and not token F1).
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(TrueCaser, self).__init__(vocab, text_field_embedder, encoder, initializer, regularizer)
        self.metrics["f1"] = F1Measure(positive_label=self.vocab.get_token_index("U", namespace="labels"))

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {}
        for metric_name, metric in self.metrics.items():
            if "f1" in metric_name:
                p,r,f1 = metric.get_metric(reset)
                metrics_to_return["p"] = p
                metrics_to_return["r"] = r
                metrics_to_return["f1"] = f1
            else:
                metrics_to_return[metric_name] = metric.get_metric(reset)

        return metrics_to_return
