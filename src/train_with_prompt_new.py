from typing import Dict, Any

import os
import random
import pickle
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import Dataset
import argparse
from torch.utils.data import DataLoader

from openprompt.plms import load_plm
from openprompt.data_utils import InputExample

from sklearn.metrics import roc_auc_score

CATEGORY_CNT = 14

pd.set_option('display.max_columns', None)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--root', type=str, default='../data')
    # 指定数据集规模
    parser.add_argument('--split', type=str, default='small')
    parser.add_argument('--pretrain', type=str, default='../output_model/checkpoint-8000')
    # 如何得到news的表示，nseg相当于CLS，论文里是MRR>MEAN>NSEG
    parser.add_argument('--news_type', type=str, default='title',
                        help='title or both')
    parser.add_argument('--news_mode', type=str, default='cls',
                        help='cls, mean, max or attention')
    # average hist len = 32.301
    parser.add_argument('--hist_max_len', type=int, default=50)
    # average title len = 10.80
    # average abstract len = 33.10
    parser.add_argument('--seq_max_len', type=int, default=20)
    parser.add_argument('--score_type', type=str, default='weighted')
    # 指定模型路径
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--output', type=str, default='../output')
    parser.add_argument('--epoch', type=int, default=5)  # always set 5 in small dataset, 2 in large
    parser.add_argument('--batch_size', type=int, default=128)  # default=128, two 3090 gpus
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--distribution', action='store_true')
    parser.add_argument('--eval', action='store_true')
    # 增加随机数种子
    parser.add_argument('--seed', type=int, default=2022)
    args = parser.parse_args()
    return args

'''
def auc_func(grouped_df):
    if sum(grouped_df["label"]) == 0 or sum(grouped_df["label"]) == len(grouped_df["label"]):
        return 1.0
    return roc_auc_score(grouped_df["label"], grouped_df["score"])
'''

def set_environment(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class MindDatasetForPrompt(Dataset):
    def __init__(self,
                 root: str,
                 mode: str = 'train',
                 split: str = 'little',
                 hist_max_len: int = 20,
                 seq_max_len: int = 300,
                 data_type: str = 'title'
                 ) -> None:
        # 把MindDatasetr类的对象转化为父类Dataset的对象
        super(MindDatasetForPrompt, self).__init__()
        self.data_path = os.path.join(root, split)
        self._mode = mode
        self._split = split
        self._data_type = data_type
        self._mode = mode
        self._hist_max_len = hist_max_len
        self._seq_max_len = seq_max_len
        # 读取新闻标题，即self._examples主要是id信息，self._news主要是文本信息
        # 读取数据, dataframe
        self._examples = self.get_examples(negative_sampling=4)

    def get_examples(self,
                     negative_sampling
                     ) -> Any:
        behavior_file = os.path.join(self.data_path, self._mode, 'behaviors.tsv')
        # print(behavior_file)
        # 读取id, user_id, time, 历史点击, 曝光点击
        df = pd.read_csv(behavior_file, sep='\t', header=None,
                         names=['impression_id', 'user_id', 'time', 'news_history', 'impressions'])
        # print(df)
        CTR_file = os.path.join(self.data_path, "ctr.csv")
        CTR_dataframe = pd.read_csv(CTR_file)
        CTR = dict(zip(CTR_dataframe["news_id"], CTR_dataframe["ctr"].apply(lambda x: float(x))))

        df["impressions_id"] = df["impressions"].apply(lambda x: x.split())

        # 没有点击历史的用户直接不用于训练
        # 注释掉后，没有点击历史的用户用于训练，让热度发挥较大的作用

        # if self._mode == 'train':
        #     df = df.dropna(subset=['news_history'])
        # else:
        #     df['news_history'] = df['news_history'].fillna('')

        df['news_history'] = df['news_history'].fillna('')
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
        else:
            df["ctr"] = df["impression"].apply(lambda x: [CTR[x.split("-")[0]]])
        # print(df)
        df.drop('impression', axis=1, inplace=True)
        df.drop('impressions_id', axis=1, inplace=True)
        df.drop('impression_list', axis=1, inplace=True)
        df.drop('ctr', axis=1, inplace=True)
        # print('------dataset------')
        # print(df)
        return df

    def get_input_examples(self) -> Any:
        input_examples = []
        df = self._examples
        # print(1)
        # print(df)
        for idx, row in enumerate(df[['news_history','news_id', 'click']].values):
            # print(row)
            news_history, candidate_news_id, click = row
            news_history = news_history.replace(' ', ',')
            # print(news_history, candidate_news_id, click)
            for(i, news_id) in enumerate(candidate_news_id):
                input_examples.append(InputExample(text_a=news_history, text_b=news_id, label=click[i]))
            # print(input_examples)
        return input_examples

    def collate(self, batch: Dict[str, Any]):
        # print("collate, batch_size", len(batch))
        # batch_size * click * max_seq_len
        curr_input_ids = torch.tensor([item['curr_input_ids'] for item in batch])
        # batch_size * click * max_seq_len
        # curr_type_ids = torch.tensor([item['curr_type_ids'] for item in batch])
        # batch_size * click * max_seq_len
        # curr_input_mask = torch.tensor([item['curr_input_mask'] for item in batch])
        # batch_size * click
        # curr_category_ids = torch.tensor([item['curr_category_ids'] for item in batch])
        # batch_size * max_hist_len * max_seq_len
        hist_input_ids = torch.tensor([item['hist_news']['hist_input_ids'] for item in batch])
        # batch_size * max_hist_len * max_seq_len
        # hist_token_type = torch.tensor([item['hist_news']['hist_token_type'] for item in batch])
        # batch_size * max_hist_len * max_seq_len
        # hist_input_mask = torch.tensor([item['hist_news']['hist_input_mask'] for item in batch])
        # batch_size * max_hist_len * 1
        # hist_mask = torch.tensor([item['hist_news']['hist_mask'] for item in batch])
        # batch_size * max_hist_len
        # hist_category_ids = torch.tensor([item['hist_news']['hist_category_ids'] for item in batch])
        #
        # CTR = torch.tensor([item['CTR'] for item in batch])
        # recency
        # recency = torch.tensor([item['recency'] for item in batch])

        inputs = {'curr_input_ids': curr_input_ids,
                  # 'curr_type_ids': curr_type_ids,
                  # 'curr_input_mask': curr_input_mask,
                  # 'curr_category_ids': curr_category_ids,
                  # 'CTR': CTR,
                  # 'recency': recency,
                  'hist_input_ids': hist_input_ids,
                  # 'hist_token_type': hist_token_type,
                  # 'hist_input_mask': hist_input_mask,
                  # 'hist_mask': hist_mask,
                  # 'hist_category_ids': hist_category_ids,
                  }

        if self._mode == 'train':
            inputs['click_label'] = torch.tensor([item['click_label'] for item in batch])
            # print(inputs)
            return inputs
        elif self._mode == 'dev':
            inputs['impression_id'] = [item['impression_id'] for item in batch]
            inputs['click_label'] = torch.tensor([item['click_label'] for item in batch])
            return inputs
        elif self._mode == 'test':
            inputs['impression_id'] = [item['impression_id'] for item in batch]
        else:
            raise ValueError('Mode must be `train`, `dev` or `test`.')

    def __len__(self):
        return len(self._examples)



class MindDataLoaderForPrompt(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, num_workers=0,
                 pin_memory=False, sampler=None) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler,
            collate_fn=dataset.collate
        )


def main():
    args = parse_args()
    args.nprocs = torch.cuda.device_count()
    print(args)
    os.makedirs(args.output, exist_ok=True)
    set_environment(args.seed)

    train_set = MindDatasetForPrompt(
        args.root,
        mode='train',
        split=args.split,
        hist_max_len=args.hist_max_len,
        seq_max_len=args.seq_max_len,
        data_type=args.news_type
    )

    # ---------------------OpenPrompt---------------------
    # Step1: Define a task
    classes = [  # There are two classes in Sentiment Analysis, one for negative and one for positive
        "negative",
        "positive"
    ]
    dataset = train_set.get_input_examples()
    # print(dataset)

    # Step2: Obtain a PLM
    # "bert-base-cased" can be replaced by model_path
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", args.pretrain)
    # from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoModelForMaskedLM
    # plm = AutoModelForMaskedLM.from_pretrained('../output_model/checkpoint-8000')
    # tokenizer = AutoTokenizer.from_pretrained('../output_model/checkpoint-8000')
    # plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
    # Step 3. Define a Template
    from openprompt.prompts import ManualTemplate
    promptTemplate = ManualTemplate(
        text='User has clicked news like {"placeholder":"text_a"} before,  and he/she would {"mask"} news like {"placeholder":"text_b"}',
        tokenizer=tokenizer,
    )

    # Step 4. Define a Verbalizer
    from openprompt.prompts import ManualVerbalizer
    promptVerbalizer = ManualVerbalizer(
        classes=classes,
        label_words={
            "negative": ["dislike"],
            "positive": ["like", "prefer"],
        },
        tokenizer=tokenizer,
    )

    # Step 5. Construct a PromptModel
    from openprompt import PromptForClassification
    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    )

    use_cuda = True  # False
    if use_cuda:
        promptModel = promptModel.cuda()

    # Step 6. Define a DataLoader
    from openprompt import PromptDataLoader
    data_loader = PromptDataLoader(
        dataset=dataset,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=args.batch_size,
    )

    # Step 7. Train and inference
    # Now the training is standard
    from transformers import  AdamW, get_linear_schedule_with_warmup
    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters = [
        {'params': [p for n, p in promptModel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in promptModel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

    for epoch in range(args.epoch):
        tot_loss = 0
        for step, inputs in enumerate(data_loader):
            if use_cuda:
                inputs = inputs.cuda()
            print(inputs)
            logits = promptModel(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            if step %100 ==1:
                print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)

    # Evaluate
    valid_set = MindDatasetForPrompt(
        args.root,
        mode='dev',
        split=args.split,
        hist_max_len=args.hist_max_len,
        seq_max_len=args.seq_max_len,
        data_type=args.news_type
    )
    valid_dataset = valid_set.get_input_examples()
    # print(dataset)
    valid_data_loader = PromptDataLoader(
        dataset=valid_dataset,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=args.batch_size,
    )

    allpreds = []
    allscore = []
    alllabels = []
    for step, inputs in enumerate(valid_data_loader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = promptModel(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        # print(torch.softmax(logits, dim=-1).cpu().tolist())
        # print(torch.argmax(logits, dim=-1).cpu().tolist())
        # print(torch.softmax(logits, dim=-1).cpu()[:, 1].tolist())
        allscore.extend(torch.softmax(logits, dim=-1).cpu()[:, 1].tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    #print(alllabels)
    #print(allscore)
    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)])/len(allpreds)
    print("acc:", acc)
    print("auc:", roc_auc_score(alllabels, allscore))



if __name__ == "__main__":
    main()