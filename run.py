import random
import warnings
import numpy.random
import torch
import argparse
import os
import torch.nn as nn
from torch import distributed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, BertTokenizer
from torch.cuda import amp
from tqdm import tqdm
from time import gmtime, strftime

from src.data_loader import MindDataset
from src.model import UNBERT
from src.loss import CLLoss, cl_loss, Loss
from src.eval import dev, test
warnings.filterwarnings("ignore")


class MindDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, num_workers=0,
                 pin_memory=False, sampler=None) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler,
            collate_fn=dataset.collate,
            drop_last=True,
        )


def set_environment(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--root', type=str, default='data')
    # 指定数据集规模
    parser.add_argument('--split', type=str, default='tiny')
    parser.add_argument('--pretrain', type=str, default='pretrainedModel/BERT-base-uncased')
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
    parser.add_argument('--tem_max_len', type=int, default=60)
    parser.add_argument('--score_type', type=str, default='weighted')
    # 指定模型路径
    parser.add_argument('--restore', type=str, default=None)
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--epoch', type=int, default=8) # always set 5 in small dataset, 2 in large
    parser.add_argument('--batch_size', type=int, default=2) # default=128, two 3090 gpus
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--distribution', action='store_true')
    parser.add_argument('--eval', action='store_true')
    # 对比学习不知道干嘛用的temp
    parser.add_argument('--temp', type=float, default=0.05)
    # 对比学习模块
    parser.add_argument('--cl_category', action='store_true')
    parser.add_argument('--cl_user', action='store_false')
    parser.add_argument('--cl_news_category', action='store_false')
    parser.add_argument('--cl_news_subcategory', action='store_false')
    parser.add_argument('--user_same', type=int, default=1)
    # loss权重
    parser.add_argument('--category_weight', type=float, default=0.0)
    parser.add_argument('--user_weight', type=float, default=0.1)
    parser.add_argument('--news_category_weight', type=float, default=0.1)
    parser.add_argument('--news_subcategory_weight', type=float, default=0.1)
    # 增加随机数种子
    parser.add_argument('--seed', type=int, default=2022)
    # continuous prompt
    parser.add_argument('--num_conti1', default=2, type=int, help='number of continuous tokens')
    parser.add_argument('--num_conti2', default=2, type=int, help='number of continuous tokens')
    parser.add_argument('--num_conti3', default=2, type=int, help='number of continuous tokens')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # args.use_amp = True
    # args.eval = True
    args.nprocs = torch.cuda.device_count()
    os.makedirs(args.output, exist_ok=True)
    set_environment(args.seed)
    # todo 这部分要重构
    log_file = os.path.join(args.output, "{}-{}-{}.log".format(
                    args.mode, args.split, strftime('%Y%m%d%H%M%S', gmtime())))
    def printzzz(log):
        with open(log_file, "a") as fout:
            fout.write(log + "\n")
        print(log)

    printzzz(str(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 混合自动精度
    enable_amp = True if 'cuda' in device.type else False
    if args.use_amp and enable_amp:
        scaler = amp.GradScaler(enabled=enable_amp)
    # old answer in May
    # answer = ['no', 'yes']
    # relevance
    # answer = ['unrelated', 'related']
    # emotion
    # answer = ['boring', 'interesting']
    # action
    answer = ['no', 'yes']
    # utility
    # answer = ['bad', 'good']
    # no english answer
    # answer = ['[a0]', '[a1]']
    # vocab_size = 30522 + args.num_conti1 + args.num_conti2 + args.num_conti3
    model = UNBERT(pretrained=args.pretrain,
                 device=device,
                 args=args,
                 answer=answer,)

    if args.restore is not None and os.path.isfile(args.restore):
        printzzz("restore model from {}".format(args.restore))
        state_dict = torch.load(args.restore, map_location=torch.device('cpu'))
        st = {}
        for k in state_dict:
            if k.startswith('bert'):
                st['_model'+k[len('bert'):]] = state_dict[k]
            elif k.startswith('classifier'):
                st['_dense'+k[len('classifier'):]] = state_dict[k]
            else:
                st[k] = state_dict[k]
        model.load_state_dict(st)
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(args.pretrain)
    if args.mode == "train":
        printzzz('reading training data...')
        train_set = MindDataset(
            args.root,
            tokenizer=tokenizer,
            mode='train',
            split=args.split,
            hist_max_len=args.hist_max_len,
            seq_max_len=args.seq_max_len,
            data_type=args.news_type,
            tem_max_len=args.tem_max_len,
            num_conti1=args.num_conti1,
            num_conti2=args.num_conti2,
            num_conti3=args.num_conti3,
        )
        model.vocab_size = train_set.vocab_size
        model.user_place = train_set.user_place
        model.mask_place = train_set.mask_place
        model.candidate_place = train_set.candidate_place
        train_loader = MindDataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            # 笔记本配置不行要设置的低一些
            num_workers=4,
            pin_memory=True,
        )
        printzzz('reading dev data...')
        dev_set = MindDataset(
            args.root,
            tokenizer=tokenizer,
            mode='dev',
            split=args.split,
            hist_max_len=args.hist_max_len,
            seq_max_len=args.seq_max_len,
            data_type=args.news_type,
            tem_max_len=args.tem_max_len,
            num_conti1=args.num_conti1,
            num_conti2=args.num_conti2,
            num_conti3=args.num_conti3,
        )
        dev_loader = MindDataLoader(
            dataset=dev_set,
            batch_size=args.batch_size,
            shuffle=False,
            # 笔记本配置不行要设置的低一些
            num_workers=4,
            pin_memory=True,
        )
        loss_fn = nn.CrossEntropyLoss()
        loss_fn.to(device)
        loss_calculator = Loss(loss_fn)
        model.zero_grad()

        m_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        m_optim.zero_grad()
        m_scheduler = get_linear_schedule_with_warmup(m_optim,
                      num_warmup_steps=len(train_set)//args.batch_size*2,
                      num_training_steps=len(train_set)*args.epoch//args.batch_size)
                    
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            # loss_fn = nn.DataParallel(loss_fn)
            loss_calculator = nn.DataParallel(loss_calculator)
        printzzz("start training...")
        for epoch in range(args.epoch):
            avg_loss = 0.0
            avg_category_loss = 0.0
            avg_user_loss = 0.0
            avg_news_category_loss = 0.0
            avg_news_subcategory_loss = 0.0
            batch_iterator = tqdm(train_loader, disable=False)
            if not os.path.exists('visualize-data'):
                os.mkdir('visualize-data')
            for step, train_batch in enumerate(batch_iterator):
                # train_batch, collate的返回结果
                if args.use_amp:
                    with amp.autocast(enabled=enable_amp):
                        poly_attn, batch_score, sims_masks = model(train_batch['curr_input_ids'].to(device),
                                            train_batch['curr_type_ids'].to(device),
                                            train_batch['curr_input_mask'].to(device),
                                            train_batch['curr_category_ids'].to(device),
                                            train_batch['curr_subcategory_ids'].to(device),
                                            train_batch['hist_input_ids'].to(device),
                                            train_batch['hist_token_type'].to(device),
                                            train_batch['hist_input_mask'].to(device),
                                            train_batch['hist_mask'].to(device),
                                            train_batch['hist_category_ids'].to(device),
                                            train_batch['CTR'].to(device),
                                            train_batch['recency'].to(device),
                                            train_batch['template_ids'].to(device),
                                            train_batch['template_token_type'].to(device),
                                            train_batch['template_mask'].to(device),
                                            train_batch['click_label'].to(device),
                                            )
                        batch_loss = loss_fn(batch_score,
                                             train_batch['click_label'].to(device).view(-1).long())
                        if args.cl_category:
                            category_loss = cl_loss(sims_masks[0], sims_masks[1], device)
                            batch_loss += args.category_weight * category_loss
                            sims_masks.pop(1)
                        if args.cl_user:
                            user_loss = cl_loss(sims_masks[0], sims_masks[1], device)
                            batch_loss += args.user_weight * user_loss
                            sims_masks.pop(0)
                            sims_masks.pop(0)
                        if args.cl_news_category:
                            news_category_loss = cl_loss(sims_masks[0], sims_masks[1], device)
                            batch_loss += args.news_category_weight * news_category_loss
                            sims_masks.pop(0)
                            sims_masks.pop(0)
                        if args.cl_news_subcategory:
                            news_subcategory_loss = cl_loss(sims_masks[0], sims_masks[1], device)
                            batch_loss += args.news_subcategory_weight * news_subcategory_loss
                        # batch_loss, category_loss, user_loss = loss_calculator(batch_score,
                        #                                                                train_batch['click_label'].to(
                        #                                                                    device).view(-1).long(),
                        #                                                                sims_masks, args, epoch)
                else:
                    poly_attn, batch_score, sims_masks = model(train_batch['curr_input_ids'].to(device),
                                        train_batch['curr_type_ids'].to(device),
                                        train_batch['curr_input_mask'].to(device),
                                        train_batch['curr_category_ids'].to(device),
                                        train_batch['curr_subcategory_ids'].to(device),
                                        train_batch['hist_input_ids'].to(device),
                                        train_batch['hist_token_type'].to(device),
                                        train_batch['hist_input_mask'].to(device),
                                        train_batch['hist_mask'].to(device),
                                        train_batch['hist_category_ids'].to(device),
                                        train_batch['CTR'].to(device),
                                        train_batch['recency'].to(device),
                                        train_batch['template_ids'].to(device),
                                        train_batch['template_token_type'].to(device),
                                        train_batch['template_mask'].to(device),
                                        train_batch['click_label'].to(device),
                                        )
                    batch_loss = loss_fn(batch_score,
                                         train_batch['click_label'].to(device).view(-1).long())
                    if args.cl_category:
                        category_loss = cl_loss(sims_masks[0], sims_masks[1], device)
                        batch_loss += args.category_weight * category_loss
                        sims_masks.pop(1)
                    if args.cl_user:
                        user_loss = cl_loss(sims_masks[0], sims_masks[1], device)
                        batch_loss += args.user_weight * user_loss
                    if args.cl_news_category:
                        news_category_loss = cl_loss(sims_masks[0], sims_masks[1], device)
                        batch_loss += args.news_category_weight * news_category_loss
                        sims_masks.pop(0)
                        sims_masks.pop(0)
                    if args.cl_news_subcategory:
                        news_subcategory_loss = cl_loss(sims_masks[0], sims_masks[1], device)
                        batch_loss += args.news_subcategory_weight * news_subcategory_loss
                    # batch_loss, category_loss, user_loss = loss_calculator(batch_score,
                    #                                                                train_batch['click_label'].to(
                    #                                                                    device).view(-1).long(),
                    #                                                                sims_masks,
                    #                                                                args, epoch)

                if torch.cuda.device_count() > 1:
                    batch_loss = batch_loss.mean()
                    if args.cl_category:
                        category_loss = category_loss.mean()
                    if args.cl_user:
                        user_loss = user_loss.mean()
                    if args.cl_news_category:
                        news_category_loss = news_category_loss.mean()
                    if args.cl_news_subcategory:
                        news_subcategory_loss = news_subcategory_loss.mean()
                avg_loss += batch_loss.item()
                if args.cl_category:
                    avg_category_loss += category_loss.item()
                if args.cl_user:
                    avg_user_loss += user_loss.item()
                if args.cl_news_category:
                    avg_news_category_loss += news_category_loss.item()
                if args.cl_news_subcategory:
                    avg_news_subcategory_loss += news_subcategory_loss.item()

                if args.use_amp:
                    scaler.scale(batch_loss).backward()
                    scaler.step(m_optim)
                    scaler.update()
                else:
                    batch_loss.backward()
                    m_optim.step()
                m_scheduler.step()
                m_optim.zero_grad()
            os.rename('visualize-data', 'visualize-data-epoch' + str(epoch))

            if epoch < 2:
                # don‘t eval on the first two epochs
                printzzz("Epoch {}, Avg_loss:{:.4f}, Avg_category_loss:{:.4f}, Avg_user_loss:{:.4f}, Avg_news_category_loss:{:.4f}, Avg_news_subcategory_loss:{:.4f}".format(
                        epoch + 1, avg_loss, avg_category_loss, avg_user_loss, avg_news_category_loss, avg_news_subcategory_loss))

                continue

            if args.eval:
                if args.use_amp:
                    with amp.autocast(enabled=enable_amp):
                        auc, mrr, ndcg5, ndcg10 = dev(model, dev_loader, device, args.output, epoch)
                else:
                    auc, mrr, ndcg5, ndcg10 = dev(model, dev_loader, device, args.output, epoch)
                # 小数点后取4位
                printzzz("Epoch {}, Avg_loss:{:.4f}, Avg_category_loss:{:.4f}, Avg_user_loss:{:.4f}, Avg_news_category_loss:{:.4f}, Avg_news_subcategory_loss:{:.4f}, AUC: {:.4f}, MRR: {:.4f}, NDCG@5: {:.4f}, NDCG@10: {:.4F}"
                    .format(epoch + 1, avg_loss, avg_category_loss, avg_user_loss, avg_news_category_loss, avg_news_subcategory_loss, auc, mrr, ndcg5, ndcg10))

            else:
                printzzz("Epoch {}, Avg_loss:{:.4f}".format(epoch+1, avg_loss))
            final_path = os.path.join(args.output, "epoch_{}.bin".format(epoch+1))
            if args.save:
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), final_path)
                else:
                    torch.save(model.state_dict(), final_path)
        printzzz("train success!")
    elif args.mode == "dev":
        # 这部分纯推理，因此要声明restore
        printzzz('reading dev data...')
        dev_set = MindDataset(
            args.root,
            tokenizer=tokenizer,
            mode='dev',
            split=args.split,
            hist_max_len=args.hist_max_len,
            tem_max_len=args.tem_max_len,
            num_conti1=args.num_conti1,
            num_conti2=args.num_conti2,
            num_conti3=args.num_conti3,
        )
        dev_loader = MindDataLoader(
            dataset=dev_set,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        if args.use_amp:
            with amp.autocast(enabled=enable_amp):
                auc, mrr, ndcg5, ndcg10 = dev(model, dev_loader, device, args.output)
        else:
            auc, mrr, ndcg5, ndcg10 = dev(model, dev_loader, device, args.output)
        printzzz("dev AUC: {:.4f} MRR: {:.4f} NDCG@5: {:.4F} NDCG@10: {:.4f}".format(auc, mrr, ndcg5, ndcg10))
        printzzz("dev success!")
    else:
        printzzz('reading test data...')
        test_set = MindDataset(
            args.root,
            tokenizer=tokenizer,
            mode='test',
            split=args.split,
            hist_max_len=args.hist_max_len,
            seq_max_len=args.seq_max_len,
            num_conti1=args.num_conti1,
            num_conti2=args.num_conti2,
            num_conti3=args.num_conti3,
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        test(model, test_loader, device, args.output)
        printzzz("test success!")


if __name__ == "__main__":
    main()
