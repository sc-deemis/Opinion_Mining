from config_parser import Config
import tensorflow as tf
import data_helpers
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
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
import re
confi = Config("config/system.config")
checkpoint_dir = str(confi.getConfig("VARIABLES", "checkpoint_dir"))

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

def word2vec(word):
    return model[word.lower()]


def get_legit_word(str, flag):
    if flag == 0:
        for word in reversed(str):
            if word in [".", "!"]:
                return invalid_word
            if data_helpers.is_word(word):
                return word
        return invalid_word

    if flag == 1:
        for word in str:
            if word in [".", "!"]:
                return invalid_word
            if data_helpers.is_word(word):
                return word
        return invalid_word


def get_sentences(text):
    indices = []
    for start, end in PunktSentenceTokenizer().span_tokenize(text):
        indices.append((start, end))
    return indices


def get_tokens(words):
    valid_words = []
    for word in words:
        if data_helpers.is_word(word) and word in model.wv.vocab:
            valid_words.append(word)
    return valid_words


def get_left_word(message, start):
    i = start - 1
    is_space = 0
    str = ""
    while i > -1:
        if message[i].isspace() and is_space == 1 and str.strip():
            break
        if message[i].isspace() and is_space == 1 and not data_helpers.is_word(str):
            is_space = 0
        if message[i].isspace():
            is_space = 1
        str += message[i]
        i -= 1
    str = str[::-1]
    return tokenizer.tokenize(str)


def get_right_word(message, start):
    i = start
    is_space = 0
    str = ""
    while i < len(message):
        if message[i].isspace() and is_space == 1 and str.strip():
            break
        if message[i].isspace() and is_space == 1 and not data_helpers.is_word(str):
            is_space = 0
        if message[i].isspace():
            is_space = 1
        str += message[i]
        i += 1
    return tokenizer.tokenize(str)

np.random.seed(3)
pivot = 2 * sequence_length + 1
pos_vec = np.random.uniform(-1, 1, (pivot + 1, distance_dim))
# pos_vec_entities = np.random.uniform(-1, 1, (4, FLAGS.distance_dim))

# beginning and end of sentence embeddings
beg_emb = np.random.uniform(-1, 1, embedding_size)
end_emb = np.random.uniform(-1, 1, embedding_size)
extra_emb = np.random.uniform(-1, 1, embedding_size)

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


def generate_vector(message, start1, end1, start2, end2):
    try:
        sent = get_sentences(message)
        beg = -1
        for l, r in sent:
            if (start1 >= l and start1 <= r) or (end1 >= l and end1 <= r) or (start2 >= l and start2 <= r) or (
                    end2 >= l and end2 <= r):
                if beg == -1:
                    beg = l
                fin = r

        # print(message[beg:fin])
        entity1, entity2 = message[start1:end1], message[start2:end2]
        l1 = [get_legit_word([word], 1)
              for word in tokenizer.tokenize(entity1)]
        l2 = [get_legit_word([word], 1)
              for word in tokenizer.tokenize(entity2)]

        # TODO add PCA for phrases
        temp = np.zeros(embedding_size)
        valid_words = 0
        # print(entity1)
        # print(l1)
        for word in l1:
            if word != "UNK" and data_helpers.is_word(word) and word in model.wv.vocab:
                valid_words += 1
                temp = np.add(temp, word2vec(word))
        if valid_words == 0:
            return None
        l1 = temp / float(valid_words)
        temp = np.zeros(embedding_size)
        valid_words = 0
        # print(entity2)
        # print(l2)
        for word in l2:
            if word != "UNK" and data_helpers.is_word(word) and word in model.wv.vocab:
                valid_words += 1
                temp = np.add(temp, word2vec(word))
        if valid_words == 0:
            return None
        lword1 = lword2 = rword1 = rword2 = np.zeros(50)
        l2 = temp / float(valid_words)
        if get_legit_word(get_left_word(message, start1), 0) in model.wv.vocab:
            lword1 = word2vec(
                get_legit_word(get_left_word(message, start1), 0))
        if get_legit_word(get_left_word(message, start2), 0) in model.wv.vocab:
            lword2 = word2vec(
                get_legit_word(get_left_word(message, start2), 0))
        if get_legit_word(get_right_word(message, end1), 1) in model.wv.vocab:
            rword1 = word2vec(get_legit_word(get_right_word(message, end1), 1))
        if get_legit_word(get_right_word(message, end2), 1) in model.wv.vocab:
            rword2 = word2vec(get_legit_word(get_right_word(message, end2), 1))
        # l3 = np.divide(np.add(lword1, rword1), 2.0)
        # l4 = np.divide(np.add(lword2, rword2), 2.0)
        # print(get_legit_word(get_left_word(message, start1), 0),
        #       get_legit_word(get_left_word(message, start2), 0))
        # print(get_legit_word(get_right_word(message, end1), 1),
        #       get_legit_word(get_right_word(message, end2), 1))

        # tokens in between
        l_tokens = []
        r_tokens = []
        if beg != -1:
            l_tokens = get_tokens(tokenizer.tokenize(message[beg:start1]))
        if fin != -1:
            r_tokens = get_tokens(tokenizer.tokenize(message[end2:fin]))
        in_tokens = get_tokens(tokenizer.tokenize(message[end1:start2]))
        # print(l_tokens, in_tokens, r_tokens)

        tot_tokens = len(l_tokens) + len(in_tokens) + len(r_tokens) + 2
        while tot_tokens < sequence_length:
            r_tokens.append("UNK")
            tot_tokens += 1
        # left tokens
        l_matrix = []
        l_len = len(l_tokens)
        r_len = len(r_tokens)
        m_len = len(in_tokens)
        if l_len + m_len + r_len + 2 > sequence_length:
            return None
        for idx, token in enumerate(l_tokens):
            # print(pivot + (idx - l_len), pivot + (idx - l_len - 1 - m_len))
            word_vec, pv1, pv2 = word2vec(token), pos_vec[pivot + (idx - l_len)], pos_vec[
                pivot + (idx - l_len - 1 - m_len)]
            l_matrix.append([word_vec, pv1, pv2])

        # middle tokens
        in_matrix = []
        for idx, token in enumerate(in_tokens):
            # print(idx + 1, idx - m_len)
            word_vec, pv1, pv2 = word2vec(
                token), pos_vec[idx + 1], pos_vec[idx - m_len + pivot]
            in_matrix.append([word_vec, pv1, pv2])

        # right tokens
        r_matrix = []
        for idx, token in enumerate(r_tokens):
            if token == "UNK":
                # print(idx + m_len + 2, idx + 1)
                word_vec, pv1, pv2 = extra_emb, pos_vec[
                    idx + m_len + 2], pos_vec[idx + 1]
                r_matrix.append([word_vec, pv1, pv2])
            else:
                # print(idx + m_len + 2, idx + 1)
                word_vec, pv1, pv2 = word2vec(
                    token), pos_vec[idx + m_len + 2], pos_vec[idx + 1]
                r_matrix.append([word_vec, pv1, pv2])

        tri_gram = []
        llen = len(l_matrix)
        mlen = len(in_matrix)
        rlen = len(r_matrix)
        dist = llen + 1
        if llen > 0:
            if llen > 1:
                tri_gram.append(
                    np.hstack((beg_emb, l_matrix[0][0], l_matrix[1][0], l_matrix[0][1], l_matrix[0][2])))
                for i in range(1, len(l_matrix) - 1):
                    tri_gram.append(
                        np.hstack((l_matrix[i - 1][0], l_matrix[i][0], l_matrix[i + 1][0], l_matrix[i][1],
                                   l_matrix[i][2])))
                tri_gram.append(np.hstack((l_matrix[llen - 2][0], l_matrix[llen - 1][0], l1, l_matrix[llen - 1][1],
                                           l_matrix[llen - 2][2])))
            else:
                tri_gram.append(
                    np.hstack((beg_emb, l_matrix[0][0], l1, l_matrix[0][1], l_matrix[0][2])))
            if mlen > 0:
                tri_gram.append(
                    np.hstack((l_matrix[llen - 1][0], l1, in_matrix[0][0], pos_vec[0], pos_vec[pivot - dist])))
            else:
                tri_gram.append(
                    np.hstack((l_matrix[llen - 1][0], l1, l2, pos_vec[0], pos_vec[pivot - dist])))
        else:
            if mlen > 0:
                tri_gram.append(
                    np.hstack((beg_emb, l1, in_matrix[0][0], pos_vec[0], pos_vec[pivot - dist])))
            else:
                tri_gram.append(
                    np.hstack((beg_emb, l1, l2, pos_vec[0], pos_vec[pivot - dist])))

        if mlen > 0:
            if mlen > 1:
                tri_gram.append(np.hstack(
                    (l1, in_matrix[0][0], in_matrix[1][0], in_matrix[0][1], in_matrix[0][2])))
                for i in range(1, len(in_matrix) - 1):
                    tri_gram.append(np.hstack((in_matrix[i - 1][0], in_matrix[i][0], in_matrix[i + 1][0],
                                               in_matrix[i][1], in_matrix[i][2])))
                tri_gram.append(np.hstack((in_matrix[mlen - 2][0], in_matrix[mlen - 1][0], l2,
                                           in_matrix[mlen - 1][1], in_matrix[mlen - 2][2])))
            else:
                tri_gram.append(
                    np.hstack((l1, in_matrix[0][0], l2, in_matrix[0][1], in_matrix[0][2])))
            if rlen > 0:
                tri_gram.append(np.hstack(
                    (in_matrix[mlen - 1][0], l2, r_matrix[0][0], pos_vec[dist], pos_vec[0])))
            else:
                tri_gram.append(
                    np.hstack((in_matrix[mlen - 1][0], l2, end_emb, pos_vec[dist], pos_vec[0])))
        else:
            if rlen > 0:
                tri_gram.append(
                    np.hstack((l1, l2, r_matrix[0][0], pos_vec[dist], pos_vec[0])))
            else:
                tri_gram.append(
                    np.hstack((l1, l2, end_emb, pos_vec[dist], pos_vec[0])))
        if rlen > 0:
            if rlen > 1:
                tri_gram.append(
                    np.hstack((l2, r_matrix[0][0], r_matrix[1][0], r_matrix[0][1], r_matrix[0][2])))
                for i in range(1, len(r_matrix) - 1):
                    tri_gram.append(np.hstack(
                        (r_matrix[i - 1][0], r_matrix[i][0], r_matrix[i + 1][0], r_matrix[i][1], r_matrix[i][2])))
                tri_gram.append(np.hstack((r_matrix[rlen - 2][0], r_matrix[rlen - 1][0], end_emb,
                                           r_matrix[rlen - 1][1], r_matrix[rlen - 2][2])))

            else:
                tri_gram.append(
                    np.hstack((l2, r_matrix[0][0], end_emb, r_matrix[0][1], r_matrix[0][2])))
        print("======================================")
        # lf = np.vstack((l1, l2, l3, l4))
        print(np.asarray(tri_gram).shape)
        return np.asarray(tri_gram)
    except Exception as e:
        traceback.print_exc()
        return None

def get_valid_items(items):
    try:
        message = "{}"
        drug_json = "{}"
        sideEffect_json = "{}"
        rowkey = "{}"
        flag = 0
        for item in items:
            json_text = json.loads(item)
            rowkey = json_text["row"]
            if json_text["qualifier"] == "message":
                message = json_text["value"].lower()
            if json_text["qualifier"] == "drug":
                drug_json = json_text["value"]
            if json_text["qualifier"] == "opinions":
                sideEffect_json = json_text["value"]
            if json_text["qualifier"] == "sent_flag":
                flag = json_text["value"]
        # if flag != 0:
           # return [(rowkey, None, None, None, None, None)]
        # if flag != 0 and flag is not None:
            # return [(rowkey, None, None, None, None, None)]
        drug_json_array = json.loads(drug_json)
        sideEffect_json_array = ast.literal_eval(sideEffect_json)
        if message is None or drug_json is None or sideEffect_json is None or drug_json == "null" or sideEffect_json == "null":
            return ([(rowkey, message, None, None, None, None)])
        if not len(drug_json_array) or not len(sideEffect_json_array):
            return ([(rowkey, message, None, None, None, None)])
        arr = []
        # print(drug_json, sideEffect_json)
        for drug_json in drug_json_array:
            drug_offset_start = drug_json["startNode"]["offset"]
            drug_offset_end = drug_json["endNode"]["offset"]
            for sideEffect_json in sideEffect_json_array:
                offset_arr = [m.start() for m in re.finditer(sideEffect_json, message)]
                for oa in offset_arr:
                    sideEffect_offset_start = oa
                    row = rowkey + "-" + \
                    str(drug_offset_start) + "-" + str(sideEffect_offset_start)
                    sideEffect_offset_end = oa + len(sideEffect_json)
                    arr.append(
                    (row, message, drug_offset_start, drug_offset_end, sideEffect_offset_start, sideEffect_offset_end))
        return arr
    except Exception as e:
        traceback.print_exc()
        return [(None, None, None, None, None, None)]

def filter_rows(row):
    try:
        a = row[0]
        return False
    except:
        return True

def sent_filter_rows(row):
    for i in range(len(row)):
        if row[i] is None:
            return False
    return True

def predict(rows):
    gc.collect()
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    global model
    print(
        "Loading word2vec model..................................................")
    w2v_path = str(confi.getConfig("VARIABLES", "word2vec_path"))
    model = gensim.models.Word2Vec.load(w2v_path)
    print("Word2vec model loaded........")
    graph = tf.Graph()
    #checkpoint_dir = "/tmp/tf_models/1495541425/checkpoints"
    #checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    #checkpoint_file = "/tmp/tf_models/1495541425/checkpoints/model-5000"
    print("Loading model................................")
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph(
                "{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
        print("**********************************************")
        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("X_train").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name(
            "dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]
        predictions = graph.get_operation_by_name(
            "output/predictions").outputs[0]

        # Generate batches for one epoch
        for row in rows:
            message = row[1]
            start1 = row[2]
            end1 = row[3]
            start2 = row[4]
            end2 = row[5]
            if start2 < start1:  # swap if entity2 comes first
                start1, start2 = start2, start1
                end1, end2 = end2, end1
            # print(message, start1, end1, start2, end2)
            input_vec = generate_vector(message, start1, end1, start2, end2)
            # print(input_vec)
            if input_vec is None:
                continue
            X_test = [input_vec]
            score, batch_predictions = sess.run(
                [scores, predictions], {input_x: X_test, dropout_keep_prob: 1.0})
            print(row[0], "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            yield (row[0], score[0], batch_predictions[0], message[start1:end2], message[start1:end1], message[start2:end2])

