import spacy
from spacy.training.example import Example
from spacy.util import minibatch, compounding
from spacy.tokens import DocBin

spacy.require_gpu()


def train_ner(_data_file, _output_dir, n_iter=30):
    nlp = spacy.blank("ar")
    ner = nlp.add_pipe("ner")

    # Load training data from docbin
    doc_bin = DocBin().from_disk(_data_file)
    docs = list(doc_bin.get_docs(nlp.vocab))

    # Add labels from training data to ner
    for doc in docs:
        for ent in doc.ents:
            ner.add_label(ent.label_)

    optimizer = nlp.begin_training()

    for itn in range(n_iter):
        losses = {}
        batches = minibatch(docs, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            examples = []
            for doc in batch:
                # In training, input and reference are the same docs with entities
                examples.append(Example(doc, doc))
            nlp.update(examples, drop=0.3, losses=losses, sgd=optimizer)
        print(f"Iteration {itn + 1}, Losses: {losses}")

    nlp.to_disk(_output_dir)
    print(f"Saved model to {_output_dir}")


if __name__ == "__main__":
    data_file = r"Data\train.spacy"
    output_dir = r"Models\ar_ner_cpu"
    train_ner(data_file, output_dir)
