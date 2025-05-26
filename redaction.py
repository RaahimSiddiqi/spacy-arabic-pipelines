import spacy
from presidio_analyzer import RecognizerResult  # for building the span objects
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

text = "ذهب محمد إلى جامعة بيرزيت"

model_dir = r"Models/ar_ner_trf"
nlp = spacy.load(model_dir)

entities = nlp(text).ents

# 3. Wrap as RecognizerResult
recognizer_results = [
    RecognizerResult(entity_type=entity.label_, start=entity.start_char, end=entity.end_char, score=1.0)
    for entity in entities
]

# 4. Anonymize directly using your pre-detected results
anonymizer = AnonymizerEngine()
anonymized = anonymizer.anonymize(
    text=text,
    analyzer_results=recognizer_results,
)

print("Anonymized text:", anonymized.text)
