from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import pandas as pd
import nltk
import re
import collections
pattern = re.compile(r'\s+')


def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned
        # print("data aligned",data_aligned)

    return data_aligned
def extract_phrase(data_dict):
    sen_dict = {}
    key = data_dict['input']
    key = re.sub(r'[^\w\s]', '', key)
    value = data_dict['output']
    sen_dict[key] = value
    item_dict = collections.OrderedDict()
    for i, j in sen_dict.items():
        i = i.split()
        j = j.split()
    zip_ij = zip(i,j)
    phrase_list = []

    for item in range(len(zip_ij)):
        if zip_ij[item][1]!='O':
            phrase_list.append(zip_ij[item][0])
        else:
            phrase_list.append("*")
    # for i, j in item_dict.items():
    #     if j != 'O':
    #         phrase_list.append(i)
    phrase_string = ' '.join(phrase_list)
    phrase_string=phrase_string.strip()
    phrase_list = []
    for i in phrase_string.split('*'):
        if i != ' ':
            phrase_list.append(i)
    while '' in phrase_list:
        phrase_list.remove('')
    return phrase_list


def interactive_shell(model):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info(""" """)

    # test_data = pd.read_csv("data/msg_list.csv")
    # sen_data = test_data['sentence'].values.tolist()
    input_path = "data/test.csv"
    drug_path = "input/drugs.csv"

    input_data = pd.read_csv(input_path)
    msg = input_data['sentence'].values.tolist()

    drug_data = pd.read_csv(drug_path)
    drug_data = drug_data['drugs'].values.tolist()

    # msg = msg[:10]
    msg_list = []
    for item in msg:
        item = nltk.sent_tokenize(item)
        item_list = []
        for sen in item:
            for drug in drug_data:
                sen = sen.lower()
                if drug in sen:
                    item_list.append(sen)
                    break
        item_list = ' '.join(item_list)
        msg_list.append(item_list)
    print("sent_data",msg_list)
    msg_df = pd.DataFrame()
    se = pd.Series(msg_list)
    input_data['sentence']=se.values
    input_data.to_csv("data/msg_df1.csv")
    sen_dict=collections.OrderedDict()
    for sentence in msg_list:
        print("sentence********",sentence)
        sen_data = nltk.sent_tokenize(sentence)
        print("sen_data****",sen_data)
        phrase_list=[]
        for item in sen_data:
            words_raw = item.strip().split(" ")
            words_raw = [x for x in words_raw if x]
            print("words raw****",words_raw)
            preds = model.predict(words_raw)
            to_print = align_data({"input": words_raw, "output": preds})

            print("to_print**********",to_print)

            phrase = extract_phrase(to_print)
            # phrase1 = [item for sublist in phrase for item in sublist]
            # phrase2=[x for x in phrase1 if x]
            print("phrase****",phrase)
            phrase_list.append(phrase)
    #         sen_dict = collections.OrderedDict()
        phrase_list = [item for sublist in phrase_list for item in sublist]
        sen_dict[sentence]=phrase_list
    print("sen dict******",sen_dict)
    df = pd.DataFrame()
    df['sentence']=sen_dict.keys()
    df['phrase']=sen_dict.values()
    df.to_csv("data/msg_test_phrases.csv")


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)

    # evaluate and interact
    model.evaluate(test)
    interactive_shell(model)


if __name__ == "__main__":
    main()
