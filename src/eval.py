# =========================================================================
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from typing import List, Tuple, Dict, Any

import torch
import multiprocessing
from multiprocessing import Pool
import operator
import functools
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


def flatten(lists: List[List]) -> List:
    return functools.reduce(operator.iconcat, lists, [])


def auc_func(grouped_df):
    if sum(grouped_df["label"]) == 0 or sum(grouped_df["label"]) == len(grouped_df["label"]):
        return 1.0
    return roc_auc_score(grouped_df["label"], grouped_df["score"])


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(df_groups, k=10):
    y_true = np.array(df_groups['label'])
    y_score = np.array(df_groups['score'])
    best = dcg_score(y_true, y_true, k)
    if best == 0:
        return 1.0
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(df_groups):
    y_true = np.array(df_groups['label'])
    y_score = np.array(df_groups['score'])
    if np.sum(y_true) == 0:
        return 1.0
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def calculate_single_user_metric(df_groups):
    try:
        auc = auc_func(df_groups)
        mrr = mrr_score(df_groups)
        ndcg5 = ndcg_score(df_groups, 5)
        ndcg10 = ndcg_score(df_groups, 10)
        return [auc, mrr, ndcg5, ndcg10]
    except ValueError:
        return [np.nan] * 4


def dev(model, dev_loader, device, out_path, epoch=0):
    impression_ids = []
    labels = []
    scores = []
    batch_iterator = tqdm(dev_loader, disable=False)
    for step, dev_batch in enumerate(batch_iterator):
        impression_id, click_label = dev_batch['impression_id'], dev_batch['click_label']
        with torch.no_grad():
            poly_attn, batch_score, sims_masks = model(dev_batch['curr_input_ids'].to(device),
                                dev_batch['curr_type_ids'].to(device),
                                dev_batch['curr_input_mask'].to(device),
                                dev_batch['curr_category_ids'].to(device),
                                dev_batch['curr_subcategory_ids'].to(device),
                                dev_batch['hist_input_ids'].to(device),
                                dev_batch['hist_token_type'].to(device),
                                dev_batch['hist_input_mask'].to(device),
                                dev_batch['hist_mask'].to(device),
                                dev_batch['hist_category_ids'].to(device),
                                dev_batch['CTR'].to(device),
                                dev_batch['recency'].to(device),
                                dev_batch['template_ids'].to(device),
                                dev_batch['template_token_type'].to(device),
                                dev_batch['template_mask'].to(device),
                                dev_batch['click_label'].to(device),
                                )
            batch_score = batch_score.softmax(dim=1)[:, 1]
            # batch_score = batch_score.sigmoid()
            batch_score = batch_score.detach().cpu().tolist()
            click_label = click_label.tolist()
            impression_ids.extend(impression_id)
            labels.extend(click_label)
            scores.extend(batch_score)

    labels = flatten(labels)
    # scores = flatten(scores)
    score_path = os.path.join(out_path, "dev_score_{}.tsv".format(str(epoch)))
    EVAL_DF = pd.DataFrame()
    EVAL_DF["impression_id"] = impression_ids
    EVAL_DF["label"] = labels
    EVAL_DF["score"] = scores
    EVAL_DF.to_csv(score_path, sep="\t", index=False)
    groups_iter = EVAL_DF.groupby("impression_id")
    imp, df_groups = zip(*groups_iter)
    pool = multiprocessing.Pool()
    results = pool.map(calculate_single_user_metric, df_groups)
    pool.close()
    pool.join()
    aucs, mrrs, ndcg5s, ndcg10s = np.array(results).T
    return np.nanmean(aucs), np.nanmean(mrrs), np.nanmean(ndcg5s), np.nanmean(ndcg10s)


def rank_func(x):
    scores = x["score"].tolist()
    tmp = [(i, s) for i, s in enumerate(scores)]
    tmp = sorted(tmp, key=lambda y: y[-1], reverse=True)
    rank = [(i+1, t[0]) for i, t in enumerate(tmp)]
    rank = [str(r[0]) for r in sorted(rank, key=lambda y: y[-1])]
    rank = "[" + ",".join(rank) + "]"
    return {"imp": x["impression_id"].tolist()[0], "rank": rank}


def test(model, test_loader, device, out_path):
    score_path = os.path.join(out_path, "test_score.tsv")
    outfile = os.path.join(out_path, "prediction.txt")
    impression_ids = []
    scores = []
    batch_iterator = tqdm(test_loader, disable=False)
    for step, test_batch in enumerate(batch_iterator):
        impression_id = test_batch['impression_id']
        with torch.no_grad():
            import time
            a = time.time()
            batch_score = model(test_batch['input_ids'].to(device), 
                                test_batch['input_mask'].to(device), 
                                test_batch['segment_ids'].to(device),
                                test_batch['news_segment_ids'].to(device),
                                test_batch['sentence_ids'].to(device),
                                test_batch['sentence_mask'].to(device),
                                test_batch['sentence_segment_ids'].to(device),
                                test_batch['recency'].to(device),
                                test_batch['CTR'].to(device))
            print('bbbbbbb', time.time() - a)
            batch_score = batch_score.softmax(dim=-1)[:, 1].squeeze(-1)
            batch_score = batch_score.detach().cpu().tolist()
            if not isinstance(batch_score, list):
                batch_score = [batch_score]
            impression_ids.extend(impression_id)
            scores.extend(batch_score)
            print('aaaaaaaa', time.time() - a)

    EVAL_DF = pd.DataFrame()
    EVAL_DF["impression_id"] = impression_ids
    EVAL_DF["score"] = scores
    EVAL_DF.to_csv(score_path, sep="\t", index=False)
    groups_iter = EVAL_DF.groupby("impression_id")
    imp, df_groups = zip(*groups_iter)
    pool = multiprocessing.Pool()
    result = pool.map(rank_func, df_groups)
    pool.close()
    pool.join()
    imps = [r["imp"] for r in result]
    ranks = [r["rank"] for r in result]
    with open(outfile, "w") as fout:
        out = [str(imp) + " " + rank for imp, rank in zip(imps, ranks)]
        fout.write("\n".join(out))
    return
