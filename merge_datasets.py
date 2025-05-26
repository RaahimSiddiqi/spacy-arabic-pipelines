import spacy
from spacy.tokens import DocBin
from pathlib import Path


def merge_spacy_files(output_path: str | Path, *input_paths: str | Path, lang: str = "ar") -> None:
    """
    Merge multiple .spacy DocBin files into one.

    Parameters
    ----------
    output_path : str | Path
        Where to write the merged file.
    *input_paths : str | Path
        One or more .spacy files to merge.
    lang : str, default "ar"
        Language code to create a blank vocab (must match your data).
    """
    nlp = spacy.blank(lang)
    merged = DocBin()

    for path in map(Path, input_paths):
        db = DocBin().from_disk(path)
        for doc in db.get_docs(nlp.vocab):
            merged.add(doc)
        print(f"✔ Added {len(db)} docs from {path.name}")

    merged.to_disk(Path(output_path))
    print(f"\n✅  Saved {len(merged)} docs to {output_path}")


# ------------------------------------------------------------------
# Example usage – adapt the paths below to your own filenames
# ------------------------------------------------------------------
train_file = r"Data\train.spacy"
test_file = r"Data\test.spacy"
out_file = r"Data\train_test.spacy"

merge_spacy_files(out_file, train_file, test_file)
