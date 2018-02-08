from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
import pandas as pd
import nltk
import re
import collections
pattern = re.compile(r'\s+')

drug_path = "input/drugs.csv"
drug_data = pd.read_csv(drug_path)
drug_data = drug_data['drugs'].values.tolist()
msg = "Becca 57 and anyone else, Hi ladies, I may be at a crossroads and it may be time to change meds. need your experience and your input. please. Here's the history:I had Stage II BC in 2002. Was diagnosed Stage IV nine years later with mets to sternum and lungs. Xeloda worked for approx. a year to reduce tumors greatly, then cancer growthTamoxifen worked for 10 mo or so and got me to NED, then had some cancer activity. Afinitor and Aromasin combo worked for approx 15 mo and got me to NED again. My tumor markers kept rising but scans were still clear. So my onc and I (after much discussion) decided to leave me on Afinitor and Aromasin because it may still be working and just add Faslodex to the mix. I've had 5 Faslodex injections total (3 months on Faslodex) Now, my tumor markers are rising again. They rose a lot this last month. almost 30 points. I've heard Faslodex can take a long time before it lowers TMs. My onc even said it can take a while. He said I should do 2 more injections and then a scan. If we see cancer growth, he feels I should do IV chemo (most likely Doxil) next. I've pushed off IV chemo for 3 1/2 years now and really am not anxious to start it. I did all that with my Stage II cancer long ago. But, I know eventually, it is probably what I will have to do. I've also heard of ladies who have tried all the hormonals first before switching to IV chemo. I've only been on Tamoxifen, Aromasin, and Faslodex. My onc feels like if this current one isn't working, it probably is time to switch to a chemo. What is your experience? Have any of you had success with Faslodex and how long before it worked? I don't want to give up on a treatment too early. Anyone had Faslodex NOT work? How long did you try? Did any of you go through MORE hormonals before going to IV chemo? Or, do you think it is better to switch gears if the cancer is growing and hit it with chemo for a while? Any of your experiences will help me. Also, anyone on Doxil? What do you think? What is it like? I'm hoping and praying that with a little extra time the Faslodex will work, but would love to hear your thoughts just in case! Thank you all so much! Julie"


def align_data(data):
    """Given dict with lists, creates aligned strings
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


def interactive_shell(model,drug_data,msg):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info(""" """)
    item = nltk.sent_tokenize(msg)
    item_list = []
    for sen in item:
        for drug in drug_data:
            sen = sen.lower()
            if drug in sen:
                item_list.append(sen)
                break
    phrase_list = []
    for sentence in item_list:
        sen_data = nltk.sent_tokenize(sentence)
        for item in sen_data:
            words_raw = item.strip().split(" ")
            words_raw = [x for x in words_raw if x]
            print("words raw****",words_raw)
            preds = model.predict(words_raw)
            to_print = align_data({"input": words_raw, "output": preds})
            phrase = extract_phrase(to_print)
            phrase_list.append(phrase)
    phrase_list = [item for sublist in phrase_list for item in sublist]
    phrase_list = list(set(phrase_list))
    return phrase_list


def main():
    # create instance of config
    config = Config()

    # build model
    # model = NERModel(config)
    # model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test  = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)

    # evaluate and interact
    model.evaluate(test)
    interactive_shell(model,drug_data,msg)


if __name__ == "__main__":
    main()
