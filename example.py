from allennlp.predictors.predictor import Predictor
from mylib import *
from allennlp.models.archival import load_archive

archive = load_archive("wiki-truecaser-model.tar.gz")
predictor = Predictor.from_archive(archive, "truecaser-predictor")

out= predictor.predict("jared smith lives in paris .")
outline = predictor.dump_line(out)

print(outline)
