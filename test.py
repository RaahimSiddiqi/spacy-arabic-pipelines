import spacy


def demo(model_dir, sentences):
    nlp = spacy.load(model_dir)

    for sent in sentences:
        doc = nlp(sent)
        print(f"\n📝  Sentence: {sent}")
        if doc.ents:
            for ent in doc.ents:
                print(f"  • {ent.text:<25}  → {ent.label_}  (start={ent.start_char}, end={ent.end_char})")
        else:
            print("  • No entities found")


if __name__ == "__main__":
    model_path = r"Models\ar_ner_cpu"  # adjust if needed

    sample_sentences = [
        "تخرج عبد الغني المجدلي من جامعة بيرزيت عام 1972.",
        "اجتمع رئيس بلدية البيرة مع الهيئة الإسلامية في القدس.",
        "سيعقد المؤتمر يوم 12/5/2025 في مدينة الرياض."
    ]

    demo(model_path, sample_sentences)
