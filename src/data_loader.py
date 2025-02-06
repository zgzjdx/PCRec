from typing import Dict, Any

import os
import random
import pickle
import numpy as np
import pandas as pd
import json
import torch
from nltk import data
data.path.append(os.path.join("..", "..", "..", "data", "nltk_data"))
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from transformers import AutoTokenizer


CATEGORY_CNT = 14

stop_words = set(stopwords.words('english'))
word_tokenizer = RegexpTokenizer(r'\w+')
pd.set_option('display.max_columns', None)


def remove_stopword(sentence):
    return ' '.join([word for word in word_tokenizer.tokenize(sentence) if word not in stop_words])


def sampling(imps, ratio=4):
    pos = []
    neg = []
    res = ""
    for imp in imps.split():
        # 如果点击就进pos, 没有就进neg
        if imp[-1] == '1':
            pos.append(imp)
        else:
            neg.append(imp)
    n_neg = ratio
    if n_neg > len(neg):
        for idx in range(n_neg - len(neg)):
            neg.append(random.choice(neg))
    # sample后，len(neg) == 4 * len(pos)

    for p in pos:
        sample_neg = random.sample(neg, n_neg)
        tmp_res = [p] + sample_neg
        random.shuffle(tmp_res)
        if res == "":
            res = " ".join(tmp_res)
        else:
            res = res + '\t' + " ".join(tmp_res)
    # 输入的是一个人的impresson，里面对每个新闻记录了点击or没有点击，输出的res是一个个用 '\t'隔开的单元，每个单元内有一个正样本，4个负样本
    # random.shuffle(neg)
    # res = pos + neg
    # random.shuffle(res)
    return res


def get_recency_dict(x, y):
    return dict(zip(x.split(" "), y))


def get_recency(x, y):
    return [x[imp] for imp in y.split()]


def process_click(imps):
    click = []
    for imp in imps.split():
        imp_list = imp.split('-')
        click.append(int(imp_list[1]))
    return click


def process_news_id(imps):
    news_id = []
    for imp in imps.split():
        imp_list = imp.split('-')
        news_id.append(imp_list[0])
    return news_id


class MindDataset(Dataset):
    def __init__(self,
                 root: str,
                 tokenizer: AutoTokenizer,
                 mode: str = 'train',
                 split: str = 'small',
                 hist_max_len: int = 20,
                 seq_max_len: int = 300,
                 data_type: str = 'title'
                 ) -> None:
        # 把MindDatasetr类的对象转化为父类Dataset的对象
        super(MindDataset, self).__init__()
        self.data_path = os.path.join(root, split)
        self._mode = mode
        self._split = split
        self._data_type = data_type
        self._tokenizer = tokenizer
        self._mode = mode
        self._hist_max_len = hist_max_len
        self._seq_max_len = seq_max_len
        # 读取新闻标题，即self._examples主要是id信息，self._news主要是文本信息
        self.category2id = self.read_json('category2id.json')
        self.subcategory2id = self.read_json('subcategory2id.json')
        self._news = self.process_news()
        # 读取数据, dataframe
        self._examples = self.get_examples(negative_sampling=4)

    def get_examples(self,
                     negative_sampling 
                     ) -> Any:
        behavior_file = os.path.join(self.data_path, self._mode, 'behaviors.tsv')
        # 读取id, user_id, time, 历史点击, 曝光点击
        df = pd.read_csv(behavior_file, sep='\t', header=None,
                         names=['impression_id', 'user_id', 'time', 'news_history', 'impressions'])
        CTR_file = os.path.join(self.data_path, "ctr.csv")
        CTR_dataframe = pd.read_csv(CTR_file)
        CTR = dict(zip(CTR_dataframe["news_id"], CTR_dataframe["ctr"].apply(lambda x: float(x))))
        recency_file = os.path.join(self.data_path, self._mode, "recency.csv")
        recency_df = pd.read_csv(recency_file)
        df["recency"] = recency_df["recency"].apply(lambda x: list(map(int, str(x).split(", "))))
        df["recency_dict"] = list(map(lambda x, y: get_recency_dict(x, y), df["impressions"], df["recency"]))
        # dict(zip(x["impressions"].split(" "), x["recency"]))
        df["impressions_id"] = df["impressions"].apply(lambda x: x.split())
        # todo 缺失值可以想办法填充，这是用户的兴趣，很重要
        # 没有点击历史的用户直接不用于训练
        # 注释掉后，没有点积历史的用户用于训练，让热度发挥较大的作用
        if self._mode == 'train':
            df = df.dropna(subset=['news_history'])
        else:
            df['news_history'] = df['news_history'].fillna('')
        #df['news_history'] = df['news_history'].fillna('')
        # 负采样,最后的impression中有多个单元间隔'\t'组成，每个单元是一个正样本和四个负样本
        if self._mode == 'train' and negative_sampling is not None:
            df['impressions'] = df['impressions'].apply(lambda x: sampling(
                x, ratio=negative_sampling))
        # 把若干个impression都拆开来，比如user1有4个impression，就一行变成4行
        # 实际上是一个单元变成一行，一行中有一个正样本四个负样本
        # todo 这里有很多重复计算的，user1的embedding就重复计算了四次
        if self._mode == 'train':
            df = df.drop('impressions', axis=1).join(df['impressions'].str.split('\t', expand=True).stack().
                                                     reset_index(level=1, drop=True).rename('impression'))
            df = df.reset_index(drop=True)
        else:
            df = df.drop('impressions', axis=1).join(df['impressions'].str.split(' ', expand=True).stack().
                                                     reset_index(level=1, drop=True).rename('impression'))
            df = df.reset_index(drop=True)
        # news_id = label_id, click = label
        if self._mode == 'test':
            df['news_id'] = df['impression']
            df['click'] = [-1] * len(df)
        else:
            df['news_id'] = df['impression'].apply(lambda x: process_news_id(x))
            df['click'] = df['impression'].apply(lambda x: process_click(x))
        # 处理后的newsid列是5个新闻的id
        # 处理后的click列是5个新闻被点击的情况 0 0 0 1 0 
        df["impression_list"] = df["impression"].apply(lambda x: x.split())
        if self._mode == "train":
            df["ctr"] = df["news_id"].apply(lambda x: [CTR[t] for t in x]) 
            df["recency"] = list(map(lambda x, y: get_recency(x, y), df["recency_dict"], df["impression"]))
        else:
            df["ctr"] = df["impression"].apply(lambda x: [CTR[x.split("-")[0]]])
            df["recency"] = list(map(lambda x, y: get_recency(x, y), df["recency_dict"], df["impression"]))
        
        # diversity_path = os.path.join("data", self._split, "diversity_score.json")
        # with open(diversity_path, 'r', encoding='utf-8') as f:
        #     diversity = json.load(f)
        #
        # diversity_list = [diversity[user] for user in df['user_id']]
        #
        # df['diversity_score'] = diversity_list
        return df

    def process_news(self) -> Dict[str, Any]:
        # 处理新闻， 从文字变为idx
        filepath = os.path.join(self.data_path, 'news_dict.pkl')
        if os.path.exists(filepath):
            print('Loading news info from', filepath)
            with open(filepath, 'rb') as fin: news = pickle.load(fin)
            return news
        news = dict()
        # news = dict{news_id{title, }, ...}
        news = self.read_news(news, os.path.join(self.data_path, 'train'))
        news = self.read_news(news, os.path.join(self.data_path, 'dev'))
        # if self._split == 'large':
            # news = self.read_news(news, os.path.join(self.data_path, 'test'))
        print('Saving news info from', filepath)
        with open(filepath, 'wb') as fout:
            pickle.dump(news, fout)
        return news

    def read_news(self,
                  news: Dict[str, Any],
                  filepath: str,
                  drop_stopword: bool = False,
                  ) -> Dict[str, Any]:
        with open(os.path.join(filepath, 'news.tsv'), encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            # id 大类(如health) 小类(如voices) title abstract(很短的abstract) url title_entities abstract_entities
            splitted = line.strip('\n').split('\t')
            news_id = splitted[0]
            if news_id in news:
                continue
            category = splitted[1].lower()
            sub_category = splitted[2].lower()
            title = splitted[3].lower()
            abstract = splitted[4].lower()
            if drop_stopword:
                title = remove_stopword(title)
                abstract = remove_stopword(abstract)
            news[news_id] = dict()
            news[news_id]['category'] = self.category2id[category]
            # news[news_id]['subcategory'] = self.subcategory2id[sub_category]
            title_words = self._tokenizer.tokenize(title)
            news[news_id]['title'] = self._tokenizer.convert_tokens_to_ids(title_words)
            abstract_words = self._tokenizer.tokenize(abstract)
            news[news_id]['abstract'] = self._tokenizer.convert_tokens_to_ids(abstract_words)
        return news

    def collate(self, batch: Dict[str, Any]):
        # print("collate, batch_size", len(batch))
        # batch_size * click * max_seq_len
        curr_input_ids = torch.tensor([item['curr_input_ids'] for item in batch])
        # batch_size * click * max_seq_len
        curr_type_ids = torch.tensor([item['curr_type_ids'] for item in batch])
        # batch_size * click * max_seq_len
        curr_input_mask = torch.tensor([item['curr_input_mask'] for item in batch])
        # batch_size * click
        curr_category_ids = torch.tensor([item['curr_category_ids'] for item in batch])
        # batch_size * max_hist_len * max_seq_len
        hist_input_ids = torch.tensor([item['hist_news']['hist_input_ids'] for item in batch])
        # batch_size * max_hist_len * max_seq_len
        hist_token_type = torch.tensor([item['hist_news']['hist_token_type'] for item in batch])
        # batch_size * max_hist_len * max_seq_len
        hist_input_mask = torch.tensor([item['hist_news']['hist_input_mask'] for item in batch])
        # batch_size * max_hist_len * 1
        hist_mask = torch.tensor([item['hist_news']['hist_mask'] for item in batch])
        # batch_size * max_hist_len
        hist_category_ids = torch.tensor([item['hist_news']['hist_category_ids'] for item in batch])
        # 
        CTR = torch.tensor([item['CTR'] for item in batch])
        # recency
        recency = torch.tensor([item['recency'] for item in batch])
        # diversity
        # diversity_score=torch.tensor([item['diversity_score'] for item in batch])

        inputs = {'curr_input_ids': curr_input_ids,
                  'curr_type_ids': curr_type_ids,
                  'curr_input_mask': curr_input_mask,
                  'curr_category_ids': curr_category_ids,
                  'CTR': CTR,
                  'recency': recency,
                  # 'diversity_score': diversity_score,
                  'hist_input_ids': hist_input_ids,
                  'hist_token_type': hist_token_type,
                  'hist_input_mask': hist_input_mask,
                  'hist_mask': hist_mask,
                  'hist_category_ids': hist_category_ids,
                  }

        if self._mode == 'train':
            inputs['click_label'] = torch.tensor([item['click_label'] for item in batch])
            return inputs
        elif self._mode == 'dev':
            inputs['impression_id'] = [item['impression_id'] for item in batch]
            inputs['click_label'] = torch.tensor([item['click_label'] for item in batch])
            return inputs
        elif self._mode == 'test':
            inputs['impression_id'] = [item['impression_id'] for item in batch]
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def pack_bert_features(self, example: Any):
        # todo 考虑新闻影响力，这样没有history的也能做
        # 要预测是否点击的新闻
        # 考虑CLS和SEP占据的3个token
        # print(example)
        if self._data_type == 'title':
            curr_input_ids = [self._news[x]['title'] for x in example['news_id']]
        else:
            curr_input_ids = [self._news[x]['title'] + self._news[x]['abstract'] for x in example['news_id']]

        curr_input_ids = [[self._tokenizer.cls_token_id] + x[:self._seq_max_len - 2] + [self._tokenizer.sep_token_id]
                          for x in curr_input_ids]
        curr_category_ids = [self._news[x]["category"]  for x in example["news_id"]]
        # current_input_ids 是一个列表，其中每个元素是一个新闻，新闻用tokenid表示，并且限制了最大长度
        curr_token_type = [[0] * len(x) for x in curr_input_ids]
        curr_input_mask = [[1] * len(x) for x in curr_input_ids]
        curr_padding_len = [self._seq_max_len - len(x) for x in curr_input_ids]
        # 对current_input_ids 中的每个元素都做补全处理，用于padding的token也有独特的id
        curr_input_ids = [x + [self._tokenizer.pad_token_id] * curr_padding_len[idx]
                          for idx, x in enumerate(curr_input_ids)]
        # [0, 0, 0]
        curr_token_type = [x + [0] * curr_padding_len[idx] for idx, x in enumerate(curr_token_type)]
        # [0, 0, 1]
        curr_input_mask = [x + [0] * curr_padding_len[idx] for idx, x in enumerate(curr_input_mask)]
        

        hist_news = {
            'hist_input_ids': [],
            'hist_token_type': [],
            'hist_input_mask': [],
            'hist_mask': [],
            'hist_category_ids': [],
        }
        # [0, 1, 2, 3, 4, 5]
        # 取user_history, 数量限制为<=hist_max_len
        # user_history是用于建模用户兴趣的，而impression是用于预测的
        for i, ns in enumerate(example['news_history'].split()[:self._hist_max_len]):
            # history每个news的长度限制<=news_max_len，考虑sep和cls占据的3个token
            if self._data_type == 'title':
                hist_input_ids = self._news[ns]['title']
            else:
                hist_input_ids = self._news[ns]['abstract']
            hist_category_ids = self._news[ns]["category"]
            hist_input_ids = hist_input_ids[:self._seq_max_len - 2]
            hist_input_ids = [self._tokenizer.cls_token_id] + hist_input_ids + [self._tokenizer.sep_token_id]
            hist_token_type = [0] * len(hist_input_ids)
            hist_input_mask = [1] * len(hist_input_ids)
            hist_padding_len = self._seq_max_len - len(hist_input_ids)
            hist_input_ids = hist_input_ids + [self._tokenizer.pad_token_id] * hist_padding_len
            hist_token_type = hist_token_type + [0] * hist_padding_len
            hist_input_mask = hist_input_mask + [0] * hist_padding_len
            assert len(hist_input_ids) == len(hist_token_type) == len(hist_input_mask)
            hist_news['hist_input_ids'].append(hist_input_ids)
            hist_news['hist_token_type'].append(hist_token_type)
            hist_news['hist_input_mask'].append(hist_input_mask)
            hist_news['hist_mask'].append(np.int32(1))
            hist_news["hist_category_ids"].append(hist_category_ids)
        hist_padding_num = self._hist_max_len - len(hist_news['hist_input_ids'])
        for idx in range(hist_padding_num):
            # 如果用户的点击历史不够多，就人为的添加一些以补全
            hist_news['hist_input_ids'].append([self._tokenizer.pad_token_id] * self._seq_max_len)
            hist_news['hist_token_type'].append([0] * self._seq_max_len)
            hist_news['hist_input_mask'].append([0] * self._seq_max_len)
            hist_news['hist_mask'].append(np.int32(0))
            hist_news['hist_category_ids'].append(np.int32(CATEGORY_CNT))
        # curr_input: 用户用于预测的新闻
        # hist_news：用户用于建模兴趣的新闻
        return curr_input_ids, curr_token_type, curr_input_mask, curr_category_ids, hist_news

    def read_json(self, file_name):
        file_path = os.path.join(self.data_path, file_name)
        with open(file_path, 'r', encoding='utf-8') as fa:
            file_dict = json.load(fa)
            return file_dict

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples.iloc[index]
        ctr = self._examples.iloc[index]["ctr"]
        #
        recency = self._examples.iloc[index]["recency"]

        # diversity_score=self._examples.iloc[index]['diversity_score']
        #
        curr_input_ids, curr_token_type, curr_input_mask, curr_category_ids, hist_news = self.pack_bert_features(example)
        inputs = {'curr_input_ids': curr_input_ids,
                  'curr_type_ids': curr_token_type,
                  'curr_input_mask': curr_input_mask,
                  'curr_category_ids': curr_category_ids,
                  'CTR': ctr,
                  'recency': recency,
                  # 'diversity_score': diversity_score,
                  'hist_news': hist_news,
                  }
        if self._mode == 'train':
            # example['click'] numpy.int32
            inputs['click_label'] = example['click']
            return inputs
        elif self._mode == 'dev':
            inputs['impression_id'] = example['impression_id']
            inputs['click_label'] = example['click']
            return inputs
        elif self._mode == 'test':
            inputs['impression_id'] = example['impression_id']
            return inputs
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def __len__(self):
        return len(self._examples)
