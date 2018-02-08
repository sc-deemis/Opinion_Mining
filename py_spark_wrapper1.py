from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
import tensorflow as tf
from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
from config_parser import Config
from sent_utils import *
import data_helpers
import pandas as pd
import nltk
import re
import collections
import json
import os
import tensorflow as tf
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
import gensim
import numpy as np
import gc
import traceback
import sys
import ast

confi = Config("config/system.config")
checkpoint_dir = str(confi.getConfig("VARIABLES", "checkpoint_dir"))

os.environ[
    'PYSPARK_PYTHON'] = '/opt/cloudera/parcels/Anaconda/envs/tf1.3/bin/python'

checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
print(checkpoint_file, "++++++++++++++++++++")
model = ""
eval_train = False
distance_dim = 5
embedding_size = 50
lr = 0.0001
allow_soft_placement = True
log_device_placement = False
filter_sizes = "3,4,5"
num_filters = 256
dropout_keep_prob = 0.5
num_epochs = 1000
l2_reg_lambda = 0.0
sequence_length = 100
K = 4
early_threshold = 0.5


tokenizer = TweetTokenizer()
invalid_word = "UNK"





pattern = re.compile(r'\s+')

drug_path = "input/drugs.csv"
drug_data = pd.read_csv(drug_path)
drug_data = drug_data['drugs'].values.tolist()
msg = "Becca 57 and anyone else, Hi ladies, I may be at a crossroads and it may be time to change meds. need your experience and your input. please. Here's the history:I had Stage II BC in 2002. Was diagnosed Stage IV nine years later with mets to sternum and lungs. Xeloda worked for approx. a year to reduce tumors greatly, then cancer growthTamoxifen worked for 10 mo or so and got me to NED, then had some cancer activity. Afinitor and Aromasin combo worked for approx 15 mo and got me to NED again. My tumor markers kept rising but scans were still clear. So my onc and I (after much discussion) decided to leave me on Afinitor and Aromasin because it may still be working and just add Faslodex to the mix. I've had 5 Faslodex injections total (3 months on Faslodex) Now, my tumor markers are rising again. They rose a lot this last month. almost 30 points. I've heard Faslodex can take a long time before it lowers TMs. My onc even said it can take a while. He said I should do 2 more injections and then a scan. If we see cancer growth, he feels I should do IV chemo (most likely Doxil) next. I've pushed off IV chemo for 3 1/2 years now and really am not anxious to start it. I did all that with my Stage II cancer long ago. But, I know eventually, it is probably what I will have to do. I've also heard of ladies who have tried all the hormonals first before switching to IV chemo. I've only been on Tamoxifen, Aromasin, and Faslodex. My onc feels like if this current one isn't working, it probably is time to switch to a chemo. What is your experience? Have any of you had success with Faslodex and how long before it worked? I don't want to give up on a treatment too early. Anyone had Faslodex NOT work? How long did you try? Did any of you go through MORE hormonals before going to IV chemo? Or, do you think it is better to switch gears if the cancer is growing and hit it with chemo for a while? Any of your experiences will help me. Also, anyone on Doxil? What do you think? What is it like? I'm hoping and praying that with a little extra time the Faslodex will work, but would love to hear your thoughts just in case! Thank you all so much! Julie"

input_table = "parseddata_sample"
output_table = "parseddata_sample"





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
    zip_ij = zip(i, j)
    phrase_list = []

    for item in range(len(zip_ij)):
        if zip_ij[item][1] != 'O':
            phrase_list.append(zip_ij[item][0])
        else:
            phrase_list.append("*")
    # for i, j in item_dict.items():
    #     if j != 'O':
    #         phrase_list.append(i)
    phrase_string = ' '.join(phrase_list)
    phrase_string = phrase_string.strip()
    phrase_list = []
    for i in phrase_string.split('*'):
        if i != ' ':
            phrase_list.append(i)
    while '' in phrase_list:
        phrase_list.remove('')
    return phrase_list


def get_extracted_phrase(model, drug_data, msg):
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
            print("words raw****", words_raw)
            preds = model.predict(words_raw)
            to_print = align_data({"input": words_raw, "output": preds})
            phrase = extract_phrase(to_print)
            phrase_list.append(phrase)
    phrase_list = [item for sublist in phrase_list for item in sublist]
    phrase_list = list(set(phrase_list))
    return phrase_list


def get_opinions(rows):
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter)
    model.evaluate(test)
    tf.reset_default_graph()
    for data in rows:
        print
        "----------------------------------------------------------------"
        print
        data["system_id"]
        print(len(data["message"]))
        sen_and_inputx = data["message"]
        # print sen_and_inputx[0]

        phrases = get_extracted_phrase(model, drug_data, sen_and_inputx)
        print(phrases)
        yield (output_table, [data["system_id"], "ner", "opinions", str(phrases)])


def get_valid_rows(inputData):
    import json
    data = dict()
    text = inputData.split("\n")
    for row in text:
        jsonText = json.loads(row)
        if jsonText["qualifier"] == "message":
            data["message"] = jsonText["value"]
        if jsonText["qualifier"] == "to_process":
            data["to_process"] = jsonText["value"]
        data["system_id"] = jsonText["row"]
        if len(data) == 3:
            break
    """
    if "to_process" not in data:
        return ["None"]	   
    """
    return data





def transform(row):
    rowkey = row[0]
    score = row[1]
    val = row[2]
    segment = row[3]
    e1 = row[4]
    e2 = row[5]
    if val == 1:
        rel = "neutral"
    elif val == 0:
        rel = "positive"
    else:
        rel = "negative"
    cnt = 0
    l = len(rowkey)
    for i in range(l):
        if rowkey[l - i - 1] == '-' and cnt == 1:
            rowi = rowkey[:l - i - 1]
        if rowkey[l - i - 1] == '-':
            cnt += 1
    tuple1 = (
        output_table, [rowkey, "cnn_results", "confidence_score", str(score[val])])
    tuple2 = (output_table, [rowkey, "cnn_results", "relationType", rel])
    tuple3 = (input_table, [rowi, "cnn_results", "sent_flag", "1"])
    tuple4 = (output_table, [rowkey, "cnn_results", "segment", segment])
    tuple5 = (output_table, [rowkey, "cnn_results", "Entity1", e1])
    tuple6 = (output_table, [rowkey, "cnn_results", "Entity2", e2])
    return ([tuple1, tuple2, tuple3, tuple4, tuple5, tuple6])




def save_record(rdd):
    host = '172.31.34.145'
    keyConv1 = "org.apache.spark.examples.pythonconverters.StringToImmutableBytesWritableConverter"
    valueConv1 = "org.apache.spark.examples.pythonconverters.StringListToPutConverter"
    conf = {"hbase.zookeeper.quorum": host,
            "mapreduce.outputformat.class": "org.apache.hadoop.hbase.mapreduce.MultiTableOutputFormat",
            "mapreduce.job.output.key.class": "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
            "mapreduce.job.output.value.class": "org.apache.hadoop.io.Writable"}
    rdd.saveAsNewAPIHadoopDataset(keyConverter=keyConv1, valueConverter=valueConv1, conf=conf)


def run_pipeline_with_spark():
    sc = SparkContext(appName="SentimentCode", pyFiles=['/home/centos/sentiment_lstm/Bi-Lstm/util.py',
                                                        '/home/centos/sentiment_lstm/Bi-Lstm/lstm_model.py'])
    sys.path.append("/home/centos/VoCP-Engine/Relation_Extraction/config")
    sys.path.append("/home/centos/VoCP-Engine/Relation_Extraction/utils")
    sc.addPyFile("/home/centos/VoCP-Engine/Relation_Extraction/utils/data_helpers.py")
    sc.addPyFile("/home/centos/VoCP-Engine/Relation_Extraction/config/config_parser.py")
    ssc = StreamingContext(sc, 1)
    conf = {"hbase.zookeeper.quorum": "172.31.34.145", "hbase.mapreduce.inputtable": input_table}
    keyConv = "org.apache.spark.examples.pythonconverters.ImmutableBytesWritableToStringConverter"
    valueConv = "org.apache.spark.examples.pythonconverters.HBaseResultToStringConverter"
    hbaseRdd = sc.newAPIHadoopRDD(
        "org.apache.hadoop.hbase.mapreduce.TableInputFormat",
        "org.apache.hadoop.hbase.io.ImmutableBytesWritable",
        "org.apache.hadoop.hbase.client.Result",
        keyConverter=keyConv,
        valueConverter=valueConv,
        conf=conf)

    messageRdd = hbaseRdd.map(lambda x: x[1])  # message_rdd = hbase_rdd.map(lambda x:x[0]) will give only row-key
    rdd = messageRdd.map(lambda x: get_valid_rows(x))
    filterRdd = rdd.filter(lambda x: filter_rows(x))
    result = filterRdd.mapPartitions(lambda x: get_opinions(x))
    hbase_rdd = result.map(lambda x: x[1]).map(
        lambda x: x.split("\n"))  # message_rdd = hbase_rdd.map(lambda x:x[0]) will give only row-key
    data_rdd = hbase_rdd.flatMap(lambda x: get_valid_items(x))
    data_rdd = data_rdd.filter(lambda x: sent_filter_rows(x))
    # data_rdd.foreach(print)
    # data_rdd = data_rdd.mapPartitions(lambda row: get_input(row))
    # data_rdd = data_rdd.filter(lambda x: filter_rows(x))
    result = data_rdd.mapPartitions(lambda iter: predict(iter))
    result = result.flatMap(lambda x: transform(x))
    # result = filterRdd.flatMap(lambda x: x)
    save_record(result)


if __name__ == '__main__':
    run_pipeline_with_spark()
