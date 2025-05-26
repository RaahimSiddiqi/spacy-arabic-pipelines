import spacy
from spacy.tokens import DocBin
from spacy.training import Example


def bio_to_spacy_entities(tokens, tags, text):
    entities = []
    entity_start = None
    entity_end = None
    entity_label = None
    char_idx = 0

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        token_start = text.find(token, char_idx)
        token_end = token_start + len(token)
        char_idx = token_end

        if tag.startswith("B-"):
            if entity_start is not None:
                entities.append((entity_start, entity_end, entity_label))
            entity_start = token_start
            entity_end = token_end
            entity_label = tag[2:]
        elif tag.startswith("I-") and entity_label == tag[2:]:
            entity_end = token_end
        else:
            if entity_start is not None:
                entities.append((entity_start, entity_end, entity_label))
                entity_start = None
                entity_end = None
                entity_label = None

    if entity_start is not None:
        entities.append((entity_start, entity_end, entity_label))
    return entities


def read_bio_file_to_docbin(input_file, output_file):
    nlp = spacy.blank("ar")
    doc_bin = DocBin()

    total_tokens = 0
    total_sentences = 0

    with open(input_file, encoding="utf8") as f:
        tokens = []
        tags = []
        for line in f:
            line = line.strip()
            if line == "":
                if tokens:
                    text = " ".join(tokens)
                    entities = bio_to_spacy_entities(tokens, tags, text)
                    doc = nlp.make_doc(text)
                    ents = []
                    for start, end, label in entities:
                        span = doc.char_span(start, end, label=label)
                        if span is None:
                            # Sometimes char_span returns None if tokens mismatch, handle here
                            print(f"Skipping entity: {text[start:end]} due to alignment issue.")
                        else:
                            ents.append(span)
                    doc.ents = ents
                    doc_bin.add(doc)

                    total_sentences += 1
                    total_tokens += len(tokens)

                    tokens = []
                    tags = []
            else:
                splits = line.split()
                if len(splits) >= 2:
                    token, tag = splits[0], splits[1]
                    tokens.append(token)
                    tags.append(tag)
        # Add last sentence if no trailing newline
        if tokens:
            text = " ".join(tokens)
            entities = bio_to_spacy_entities(tokens, tags, text)
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in entities:
                span = doc.char_span(start, end, label=label)
                if span is None:
                    print(f"Skipping entity: {text[start:end]} due to alignment issue.")
                else:
                    ents.append(span)
            doc.ents = ents
            doc_bin.add(doc)

            total_sentences += 1
            total_tokens += len(tokens)

    doc_bin.to_disk(output_file)
    print(f"Saved {len(doc_bin)} docs to {output_file}")
    print(f"  Sentences: {total_sentences}")
    print(f"  Tokens:    {total_tokens}")


if __name__ == "__main__":
    input_file = r"Data\Wojood1_1_flat\test.txt"  # replace with your input file
    output_file = r"Data\test.spacy"
    read_bio_file_to_docbin(input_file, output_file)
