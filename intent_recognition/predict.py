import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import os

words_path = "intent_recognition/words.pkl"
with open(words_path, 'rb') as f_words:
    words = pickle.load(f_words)


class TextRCNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, output_size, dropout=0.5):
        super(TextRCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 这里batch_first=True，只影响输入和输出。hidden与cell还是batch在第2维
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_size,
                            num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2 + embedding_dim, output_size)

    def forward(self, x):
        # x :(batch, seq_len) = (163, 20)
        # [batch,seq_len,embedding_dim] -> (163, 20, 300)
        x = self.embedding(x)
        #out=[batch_size, seq_len, hidden_size*2]
        #h=[num_layers*2, batch_size, hidden_size]
        #c=[num_layers*2, batch_size, hidden_size]
        out, (h, c) = self.lstm(x)
        # 拼接embedding与bilstm
        out = torch.cat((x, out), 2)  # [batch_size, seq_len, embedding_dim + hidden_size*2]
        # 激活
        # out = F.tanh(out)
        out = F.relu(out)
        # 维度转换 => [batch_size, embedding_dim + hidden_size*2, seq_len]
        #out = torch.transpose(out, 1, 2),一维卷积是对输入数据的最后一维进行一维卷积
        out = out.permute(0, 2, 1)
        out = F.max_pool1d(out, out.size(2))
        out = out.squeeze(-1)  # [batch_size,embedding_dim + hidden_size * 2]
        out = self.dropout(out)
        out = self.fc(out)  # [batch_size, output_size]
        return out


model = TextRCNN(len(words), 300, 128, 2, 16)
model_path = os.path.join(os.getcwd(), "intent_recognition/model.h5")
model.load_state_dict(torch.load(model_path))

# from pyhanlp import HanLP
import hanlp

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)

# segment = HanLP.newSegment().enableCustomDictionaryForcing(True)
pos = HanLP['pos/pku']
tok = TaggingTokenization = HanLP['tok/fine']
tok.dict_combine = {'Vue', 'React', '条件渲染', '反恐精英', '反恐精英online'}

pos.dict_tags = {'Vue': 'kp', 'React': 'kp', '条件渲染': 'kp', '反恐精英online': 'kp', '反恐精英': 'kp'}


# 分词，需要将电影名，演员名和评分数字转为nm，nnt，ng
def sentence_segment(sentence):
    result = HanLP([sentence])

    word_nature = []
    index = 0
    for term in result['tok/fine'][0]:
        word_nature.append({'word': term, 'nature': result['pos/pku'][0][index]})
        index += 1
    print(word_nature)
    sentence_words = []
    for term in word_nature:
        print(term)
        if str(term['nature']) == 'kp':
            sentence_words.append('kp')
        # elif str(term['nature']) == 'nm':
        #     sentence_words.append('nm')
        # elif str(term['nature']) == 'ng':
        #     sentence_words.append('ng')
        # elif str(term['nature']) == 'm':
        #     sentence_words.append('x')
        else:
            sentence_words.extend(list(term['word']))
    print(sentence_words)
    return sentence_words


def bow(sentence, words, show_detail=True):
    sentence_words = sentence_segment(sentence)
    indexed = [words.stoi[t] for t in sentence_words]
    src_tensor = torch.LongTensor(indexed)
    src_tensor = src_tensor.unsqueeze(0)
    return src_tensor


def predict_class(sentence, model):
    sentence_bag = bow(sentence, words, False)
    model.eval()
    with torch.no_grad():
        outputs = model(sentence_bag)
    print('outputs:{}'.format(outputs))
    predicted_prob, predicted_index = torch.max(F.softmax(outputs, 1), 1)  #预测最大类别的概率与索引
    print('softmax_prob:{}'.format(predicted_prob))
    print('softmax_index:{}'.format(predicted_index))
    results = []
    #results.append({'intent':index_classes[predicted_index.detach().numpy()[0]], 'prob':predicted_prob.detach().numpy()[0]})
    results.append({'intent': predicted_index.detach().numpy()[0], 'prob': predicted_prob.detach().numpy()[0]})
    print('result:{}'.format(results))
    return results


def get_response(predict_result):
    tag = predict_result[0]['intent']
    return tag


def predict(text):
    predict_result = predict_class(text, model)
    # res = get_response(predict_result)
    # return res
    return {'intent': int(predict_result[0]['intent']), 'prob': float(predict_result[0]['prob'])}


# print(predict("Vue怎么用"))
# print(predict("反恐精英咋用"))
# print(predict("玩反恐精英要先学什么"))
# print(predict("玩反恐精英online要先学什么"))
# print(predict("反恐精英包括什么"))
# print(predict("反恐精英包括什么"))
# print(predict("反恐精英的条件渲染是什么"))
