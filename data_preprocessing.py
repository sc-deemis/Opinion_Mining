import pandas as pd
import nltk

input_path = "data/msg_list1.csv"
drug_path = "data/drug_list.csv"

input_data = pd.read_csv(input_path)
msg = input_data['sentence'].values.tolist()

drug_data = pd.read_csv(drug_path)
drug_data = drug_data['drugs'].values.tolist()

msg = msg[:10]
msg_list=[]
for item in msg:
    item = nltk.sent_tokenize(item)
    item_list=[]
    for sen in item:
        for drug in drug_data:
            sen=sen.lower()
            if drug in sen:
                item_list.append(sen)
                break
    item_list=' '.join(item_list)
    msg_list.append(item_list)

print("msg_list*******",msg_list)



