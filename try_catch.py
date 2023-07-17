import ast
import spacy
import warnings

nlp = spacy.load("ro_core_news_lg")
warnings.filterwarnings("error")  # Raise an error for warnings

with open("output.txt", "r") as f:
    lines = f.readlines()

i = 0
for line in lines:
    parsed_tuple = ast.literal_eval(line)
    parsed_string = parsed_tuple[0]
    entities = parsed_tuple[1]['entities']

    try:
        doc = nlp.make_doc(parsed_string)
        biluo_tags = spacy.training.offsets_to_biluo_tags(doc, entities)
        print("--------")
        doc = nlp(parsed_string)
        for ent in doc.ents:
            start = ent.start
            end = ent.end
            entity_text = ent.text
            entity_type = ent.label_
            print(entity_text, entity_type, start, end)

        tokens = [token.text for token in doc]

        if i == 8788:
            print(parsed_string)
            print(entities)
            print(biluo_tags)
            print(parsed_string[50: 56])
            break
        i += 1

    except UserWarning as warning:
        print(f"Warning encountered at line {i}: {warning}")
        break
