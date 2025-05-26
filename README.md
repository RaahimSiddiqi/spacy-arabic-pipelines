# spacy-arabic-pipelines
Unofficial Arabic pipelines for spacy. Initial release contains the NER pipeline, and the NER dataset using for training.
POS tagging, Lemma, and Morphology pipelines are planned.


## Data

NER Data was acquired from [SinaLab](https://github.com/SinaLab/ArabicNER) in the form of their Wojood Dataset, 
which is a corpus for Arabic nested Named Entity Recognition (NER) comprising multiple dialects, various domains (Media, History, Health, Finance, Politics, etc.)
and consisting of about 550k tokens. Further details can be found on their GitHub.

Morphology, POS, Tagger, and Lemma data will not be provided within this repo due to the size of the data. This data can
be acquired from the SinaLab repository for [sinatools](https://github.com/SinaLab/sinatools). 

## Training

```
# GPU
python -m spacy train trf_config.cfg --output ./Models/output --paths.train ./Data/train.spacy --paths.dev ./Data/val.spacy --gpu-id 0

# CPU
python -m spacy train cpu_config.cfg --output ./Models/output --paths.train ./Data/train.spacy --paths.dev ./Data/val.spacy
```

Due to large size (400mb+) of transformers based models, they are published separately in releases. 
Download them and place them in the Models folder.

## Results 

### NER


| Author            | Model              | F1 (val) | F1 (test) |
|-------------------|--------------------|----------|-----------|
| SinaLab           | AraBERTV2          | N/A      | 88.4%     |
| Ours              | Spacy (cpu)        | 86.11    | 86.4%     |
| Ours              | Spacy-Transformers | 90.02%   | 91.0%     |
| Ours** (extended) | Spacy-Transformers | 90.9%    | N/A       |

**Trained using a combination of train and test set.

### Entity-wise Results

The following table compares the F1 scores across various models we trained.

| Entity type | Spacy (CPU) | Transformers | Transformers Extended |
|-------------|-------------|--------------|-----------------------|
| PERS        | 0.8708      | 0.9442       | 0.9503                |
| ORG         | 0.8864      | 0.9289       | 0.9336                |
| DATE        | 0.9296      | 0.9465       | 0.9420                |
| OCC         | 0.8380      | 0.9018       | 0.8893                |
| FAC         | 0.7209      | 0.7979       | 0.8148                |
| LOC         | 0.7733      | 0.8212       | 0.8105                |
| GPE         | 0.8858      | 0.9290       | 0.9282                |
| NORP        | 0.6996      | 0.7401       | 0.7777                |
| ORDINAL     | 0.9210      | 0.9412       | 0.9461                |
| EVENT       | 0.7758      | 0.7842       | 0.8029                |
| CARDINAL    | 0.7653      | 0.8528       | 0.9070                |
| QUANTITY    | 0.0000      | 0.4444       | 0.4444                |
| LANGUAGE    | 0.7143      | 0.8125       | 0.7429                |
| MONEY       | 0.6190      | 0.6957       | 0.8837                |
| TIME        | 0.6667      | 0.7273       | 0.8116                |
| PRODUCT     | 0.4286      | 0.7143       | 0.5556                |
| LAW         | 0.7647      | 0.8542       | 0.8542                |
| PERCENT     | 0.7586      | 0.9600       | 1.0000                |
| WEBSITE     | 0.4658      | 0.5942       | 0.6447                |
| CURR        | 0.0000      | 0.0000       | 0.0000                |
| UNIT        | 0.0000      | 0.0000       | 0.0000                |

