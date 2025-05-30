# spacy-arabic-pipelines
Unofficial Arabic pipelines for spacy. Initial release contains the NER pipeline, and the NER dataset using for training.
POS tagging, Lemma, and Morphology pipelines are planned.


## Data

NER Data was acquired from [SinaLab](https://github.com/SinaLab/ArabicNER) in the form of their Wojood Dataset, 
which is a corpus for Arabic nested Named Entity Recognition (NER) comprising multiple dialects, various domains (Media, History, Health, Finance, Politics, etc.)
and consisting of about 550k tokens. Further details can be found on their GitHub.

Morphology, POS, Tagger, and Lemma data will not be provided within this repo due to the size of the data. This data can
be acquired from the SinaLab repository for [sinatools](https://github.com/SinaLab/sinatools). 

## Usage

```
import spacy


def demo(model_dir, sentences):
    nlp = spacy.load(model_dir)

    for sent in sentences:
        doc = nlp(sent)
        print(f"\n📝  Sentence: {sent}")
        if doc.ents:
            for ent in doc.ents:
                print(f" • {ent.text:<25}  → {ent.label_}  (start={ent.start_char}, end={ent.end_char})")
        else:
            print("  • No entities found")


if __name__ == "__main__":
    model_path = r"Models\ar_ner_cpu"  

    sample_sentences = [
        "تخرج عبد الغني المجدلي من جامعة بيرزيت عام 1972.",
        "اجتمع رئيس بلدية البيرة مع الهيئة الإسلامية في القدس.",
        "سيعقد المؤتمر يوم 12/5/2025 في مدينة الرياض."
    ]

    demo(model_path, sample_sentences)
```

 

## Training
```
# GPU
python -m spacy train ./Configs/trf_config.cfg --output ./Models/output --paths.train ./Data/train.spacy --paths.dev ./Data/val.spacy --gpu-id 0

# CPU
python -m spacy train ./Configs/cpu_config.cfg --output ./Models/output --paths.train ./Data/train.spacy --paths.dev ./Data/val.spacy
```

Due to large size (400mb+) of transformers based models, they are published separately in releases. 
Download them and place them in the Models folder.

## Results 

### NER


| Author            | Model              | F1 (val)   | F1 (test) |
|-------------------|--------------------|------------|-----------|
| SinaLab           | AraBERTV2          | N/A        | 88.4%     |
| Ours              | Spacy (cpu)        | 86.11      | 86.4%     |
| Ours              | Spacy-Transformers | 90.67%     | **91.4%** |
| Ours** (extended) | Spacy-Transformers | **91.13%** | N/A       |

**Trained using a combination of train and test set.

### Entity-wise Results

The following table compares the F1 scores across various models we trained. SinaLab metrics are from their test set, 
whereas ours are reported from the validation set.


| Entity type | SinaLab | Spacy (CPU) | Transformers | Transformers Extended  |
|-------------|---------|-------------|--------------|------------------------|
| PERS        | 0.9129  | 0.8708      | 0.9465       | 0.9542                 |
| ORG         | 0.8997  | 0.8864      | 0.9357       | 0.9360                 |
| DATE        | 0.9323  | 0.9296      | 0.9397       | 0.9448                 |
| OCC         | 0.8193  | 0.8380      | 0.9066       | 0.9103                 |
| FAC         | 0.6895  | 0.7209      | 0.7826       | 0.7938                 |
| LOC         | 0.7524  | 0.7733      | 0.8026       | 0.8533                 |
| GPE         | 0.9470  | 0.8858      | 0.9251       | 0.9298                 |
| NORP        | 0.6931  | 0.6996      | 0.7616       | 0.7656                 |
| ORDINAL     | 0.9430  | 0.9210      | 0.9505       | 0.9586                 |
| EVENT       | 0.6425  | 0.7758      | 0.8073       | 0.8103                 |
| CARDINAL    | 0.8505  | 0.7653      | 0.8776       | 0.8779                 |
| QUANTITY    | 0.2000  | 0.0000      | 0.4444       | 0.4444                 |
| LANGUAGE    | 0.8060  | 0.7143      | 0.7273       | 0.8824                 |
| MONEY       | 0.8649  | 0.6190      | 0.8085       | 0.7917                 |
| TIME        | 0.5526  | 0.6667      | 0.7246       | 0.7123                 |
| PRODUCT     | 0.2857  | 0.4286      | 0.7143       | 0.6667                 |
| LAW         | 0.8814  | 0.7647      | 0.8421       | 0.8542                 |
| PERCENT     | 0.4426  | 0.7586      | 1.0000       | 1.0000                 |
| WEBSITE     | 0.4936  | 0.4658      | 0.6533       | 0.6621                 |
| CURR        | 0.9136  | 0.0000      | 0.0000       | 0.0000                 |
| UNIT        | 0.2500  | 0.0000      | 0.0000       | 0.0000                 |


