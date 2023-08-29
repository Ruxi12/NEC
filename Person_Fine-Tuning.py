import json
import spacy
import random

import thinc
from spacy.util import minibatch, compounding
# print(spacy.__version__)
from spacy.tokens import Doc
from spacy.training import Example
from spacy.scorer import Scorer
from thinc.api import Adam
from copy import deepcopy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# for stopwords + stemming
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# download
nltk.download('stopwords')
nltk.download('punkt')
def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('romanian'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

    # Stem the tokens
    stemmer = SnowballStemmer('romanian')
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]

    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text
def parsare_text(ner_tags, tokens, space_after):
    n = len(ner_tags)
    text = ""
    i = 0  # Initialize loop index
    entity = []
    while i < n:
        # formam pe parcurs textul
        if space_after[i] == True:
            text += tokens[i] + " "
        else:
            text += tokens[i]

        if ner_tags[i].startswith("B-"):
            j = i + 1
            txt = tokens[i]
            ok = True
            if space_after[i] == True:
                txt += " "
            while j < n and ner_tags[j].startswith("I-"):
                ok = False
                if space_after[j] == True:
                    txt += tokens[j] + " "
                    text += tokens[j] + " "
                else:
                    txt += tokens[j]
                    text += tokens[j]
                j += 1

            if ok == False:  # a intrat in while - avem
                # txt = txt.strip()
                # text = text.strip()
                length_substr = len(txt)
                length = len(text)
                start = length - length_substr
                end = length

                while text[end-1] == ' ':
                    end -= 1
                entity_label = ner_tags[i][2:]
                i = j
                if entity_label == "PERSON":
                    entity.append((txt, start, end, entity_label))

            else: # este doar o singur cuvant - entitate

                new_txt = txt.strip()
                new_text = text.strip()
                length_substr = len(new_txt)
                length = len(new_text)
                start = length - length_substr
                end = length

                while text[end-1] == ' ':
                    end -= 1
                entity_label = ner_tags[i][2:]
                i = j
                if entity_label == "PERSON":
                    entity.append((txt, start, end, entity_label))
                # i += 1  # Increment loop index if no I- tag found
        else:
            i += 1  # Increment loop index for other cases
    if entity :
        return  (text, {"entities": [  (start, end, label) for (_, start, end, label) in entity]})
    return None


if __name__ == '__main__':
    # try stemming/ lemming -> stemming
    # eliminate stopwords
    # replace la diacritice
    # datele de training
    # matrice de confuzie
    with open("train.json", 'r') as file:
        json_data = json.load(file)
    training_data = []
    # gigi = "- Iohannis, Klaus Vacanță Iohannis - căruia puțin îi pasă că 40% populația țării trăiește sub limita sărăciei altfel nu ar putea să susțină sifonarea profiturilor de către multinaționalele străine și ar susține impozitul pe venit- BăsescuTraian Băsescu care a spus că aplicarea " \
    #         "unui impozit pe venit ține de „Paleoliticul Financiar” semn că joacă în continuare " \
    #         "cum îi cântă Sistemul și că i se fâlfâie de Sifonarea Profiturilor și de milioanele români "
    # print(gigi[445:452])
    for item in json_data:
        id_value = item["id"]
        ner_tags = item["ner_tags"]
        ner_ids = item["ner_ids"]
        tokens = item["tokens"]
        space_after = item["space_after"]
        data = parsare_text(ner_tags=ner_tags, tokens=tokens, space_after=space_after)
        # print(data)
        if data not in training_data and data != None:
            training_data.append(data)

    # citesc si datele din extra dataset

    # Defineste numele fisierului
    nume_fisier = "extra_dataset.json"
    # Citeste continutul fisierului
    with open(nume_fisier, "r") as fisier:
        continut = json.load(fisier)

    # train_data = (text, {"entities": [(start, end, label) for (_, start, end, label) in entity]})


    for item in continut:
        text = item["text"]
        entities = item["entities"]
        entity = []
        for elem in entities:
            start = elem["start"]
            end = elem["end"]
            label =  elem["label"]
            entity.append((start, end, label))
        data = (text, {"entities": entity})
        training_data.append(data)

    # citesc si continutul din extra_extra.json
    with open('extra_extra.json', 'r') as file:
        json_data = file.read()

    # parse the json data
    parsed_data = json.loads(json_data)

    # Extract the text and entities from the parsed data

    for elem in parsed_data:
        text = elem['text']
        entities = elem['entities']

        formatted_entities = []
        for entity in entities:
            label = entity['label']
            start = entity['start']
            end = entity['end']
            formatted_entities.append((start, end, label))

        # Create the final tuple in the desired format
        result = (text, {'entities': formatted_entities})

        training_data.append(result)

    # creare fisier cu datele parsate dupa cum le vrea Spacy
    file = open("output_file.txt", "w")
    for data in training_data:
        file.write(str(data))
        file.write("\n")

    # load the pretrained model
    nlp = spacy.load("ro_core_news_lg")

    # define the entity recognition pipeline
    ner = nlp.get_pipe("ner")
    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Disable unnecessary pipeline components
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    # citire date de validare
    with open("valid.json", 'r') as file:
        json_data = json.load(file)

    validation_data = []
    for item in json_data:
        id_value = item["id"]
        ner_tags = item["ner_tags"]
        ner_ids = item["ner_ids"]
        tokens = item["tokens"]
        space_after = item["space_after"]
        data = parsare_text(ner_tags=ner_tags, tokens=tokens, space_after=space_after)
        # print(data)
        if data not in validation_data and data != None:
            validation_data.append(data)

    # Training loop
    with nlp.disable_pipes(*unaffected_pipes):
        learn_rate = 0.0001  # Specify the learning rate
        optimizer = thinc.api.Adam(learn_rate=learn_rate)
        num_epochs = 400   #200
        losses_list = []
        patience = 12  # Number of epochs to wait for improvement before early stopping
        early_stopping_count = 0
        best_loss = float('inf')

        for iteration in range(num_epochs):
            random.shuffle(training_data)
            losses = {}
            batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))

            for batch in batches:
                texts = [text for text, _ in batch]
                annotations = [annotation for _, annotation in batch]
                examples = []

                for text, annotation in zip(texts, annotations):
                    doc = nlp.make_doc(text)
                    example = Example.from_dict(doc, annotation)
                    examples.append(example)

                nlp.update(examples, drop=0.2, sgd=optimizer, losses=losses)

            # avg_loss = losses["ner"] / len(training_data)
            avg_loss = losses["ner"]
            losses_list.append(avg_loss)

            print("Epoch:", iteration, "Loss:", avg_loss)

            # Check for early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                early_stopping_count = 0
            else:
                early_stopping_count += 1
                if early_stopping_count >= patience:
                    print("Early stopping!")
                    break
        # save the model
        trained_model_name = "/Users/ruxi.gontescu/PycharmProjects/Spacy/model"
        nlp.to_disk(trained_model_name)
        # Plot the loss curve
        plt.plot(range(len(losses_list)), losses_list)
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Training Loss Curve')
        plt.savefig('training_loss_curve.png')  # Save the plot as an image file
        plt.show()


    # Evaluate the model on validation data
    validation_losses = []
    precision_values = []
    recall_values =[]
    f1_values = []
    epochs = []
    i = 1
    for text, annotations in validation_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        loss = nlp.evaluate([example])

        # cazul ori cand nu sunt entitati in text, ori nu a gasit entitatile textului
        if loss.get('ents_p') is not None:
            epochs.append(i)
            validation_losses.append({'ents_p': loss.get('ents_p'), 'ents_r':loss.get('ents_r'), 'ents_f':loss.get('ents_f')})
        # else:
            # calculez procent de cat de corect este
            # print(len(annotations))
            # print("----------------------------")
            # precision, recall, f1 = calculate_results(annotations)
            # break
            # validation_losses.append({'ents_p': 0.0, 'ents_r': 0.0, 'ents_f': 0.0})
        # if i == 14:
        #     print("Validation on 13 ------------")
        #     print(validation_losses)

        print(i, len(validation_losses))
        # Calculate average evaluation metrics for validation data
        if loss.get('ents_p') is not None:
            avg_validation_loss = {
                'ents_p': sum(loss.get('ents_p', 0.0) for loss in validation_losses) / len(validation_losses),
                'ents_r': sum(loss.get('ents_r', 0.0) for loss in validation_losses) / len(validation_losses),
                'ents_f': sum(loss.get('ents_f', 0.0) for loss in validation_losses) / len(validation_losses)
            }
            # Print evaluation metrics
            print("Precision:", avg_validation_loss['ents_p'])
            print("Recall:", avg_validation_loss['ents_r'])
            print("F1-score:", avg_validation_loss['ents_f'])

            # store to further create the plotting
            precision_values.append(avg_validation_loss['ents_p'])
            recall_values.append(avg_validation_loss['ents_r'])
            f1_values.append(avg_validation_loss['ents_f'])
            i += 1


    # for all the model
    # Calculate average evaluation metrics for the model on the validation data
    avg_validation_loss = {
        'ents_p': sum(loss.get('ents_p', 0.0) for loss in validation_losses) / len(validation_losses),
        'ents_r': sum(loss.get('ents_r', 0.0) for loss in validation_losses) / len(validation_losses),
        'ents_f': sum(loss.get('ents_f', 0.0) for loss in validation_losses) / len(validation_losses)
    }

    # Print evaluation metrics for the model
    print("Final Precision:", sum(precision_values) / len(precision_values))
    print("Final Recall:", sum(recall_values) / len(recall_values))
    print("Final F1-score:", sum(f1_values) / len(f1_values))
    print("Pacience", patience)
    # Plotting the evolution of metrics over epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, precision_values, label='Precision')
    plt.plot(epochs, recall_values, label='Recall')
    plt.plot(epochs, f1_values, label='F1-score')
    plt.xlabel('Iteration')
    plt.ylabel('Metric Value')
    plt.title('Evolution of Precision, Recall, and F1-score over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('metrics_evolution.png')  # Save the plot as an image file
    plt.show()






