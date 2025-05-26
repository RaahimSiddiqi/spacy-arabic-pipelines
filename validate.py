# eval_dev_spacy.py
import argparse
import spacy
from spacy.scorer import Scorer
from spacy.tokens import DocBin
from spacy.training import Example

spacy.require_gpu()


def load_docbin(path, nlp):
    """Load a .spacy DocBin and return a list of Docs."""
    db = DocBin().from_disk(path)
    return list(db.get_docs(nlp.vocab))


def evaluate(model_dir: str, dev_path: str):
    nlp = spacy.load(model_dir)

    # gold docs from dev.spacy
    gold_docs = load_docbin(dev_path, nlp)

    # build Example objects containing gold + predicted docs
    examples = []
    for gold in gold_docs:
        pred = nlp(gold.text)
        examples.append(Example(predicted=pred, reference=gold))

    # one-shot scoring
    scores = Scorer().score(examples)

    print("Evaluation on dev.spacy")
    print(f"Precision: {scores['ents_p']:.3f}")
    print(f"Recall:    {scores['ents_r']:.3f}")
    print(f"F1-score:  {scores['ents_f']:.3f}")


if __name__ == "__main__":
    model = r"Models\\ar_ner_trf_extended"
    data = r"Data\test.spacy"
    evaluate(model, data)
