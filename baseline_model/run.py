# -*- coding: utf-8 -*-
import argparse
import copy, json, os

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from time import gmtime, strftime

from model.model import Baseline
from model.data import SQuAD
from model.ema import EMA
#import evaluate

def train(args, data):
    model = Baseline(args, data.WORD.vocab.vectors).to(args.device)
    
#    ema = EMA(args.exp_decay_rate)
#    for name, param in model.named_parameters():
#        if param.requires_grad:
#            ema.update(name, param.data)
#    parameters = filter(lambda p:p.requires_grad, model.parameters())
    TRG_PAD_IDX = data.WORD_DECODER.vocab.stoi[data.WORD.pad_token]
#    
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
#    
#    writer = SummaryWriter(log_dir='runs/' + args.model_time)
#    
#    model.train()
    loss, last_epoch = 0, -1
#    max_dev_exact, max_dev_f1 = -1, -1
#    
#    test_bound = 2
#    iterator = data.train_iter
#    
    for i, batch in enumerate(data.train_iter):
#        present_epoch = int(iterator.epoch)
#        if present_epoch == args.epoch:
#            break
#        
#        if present_epoch > last_epoch:
#            print('epoch:', present_epoch + 1)
#        last_epoch = present_epoch
#        
        X, attention = model(batch)
        output_dim = X.shape[-1]
        X = X.contiguous().view(-1, output_dim)
#        output_dim = X.shape[-1]
#        X = X.contiguous().view(-1, output_dim)
        label = batch.q_word_decoder[0][:, 1:].contiguous().view(-1)
#        
        optimizer.zero_grad()
        batch_loss = criterion(X, label)
        loss += batch_loss.item()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(). args.CLIP)
        optimizer.step()
#        
        print('loss: {}'.format(batch_loss))
        print('outsize: {}'.format(X.size()))
        
        
        if i == 2: break
    
    
    
    
    
#        for name, param in model.named_parameters():
#            if param.requires_grad:
#                ema.update(name, param.data)
#        
#        if (i + 1) % args.print_freq == 0:
#            dev_loss, dev_exact, dev_f1 = test(model, ema, args, data)
#            c = (i + 1) // args.print_freq
    
        
    print('end of the test')
    return model
def test():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-dim', default=8, type=int)
    parser.add_argument('--char-channel-width', default=5, type=int)
    parser.add_argument('--char-channel-size', default=100, type=int)
    
    parser.add_argument('--dev-batch-size', default=100, type=int)
    parser.add_argument('--dev-file', default='dev-v2.0.json')
    parser.add_argument('--train-batch-size', default=60, type=int)
    parser.add_argument('--train-file', default='train-v2.0.json')
    
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=12, type=int)
    
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    
    parser.add_argument('--learning-rate', default=0.5, type=float)
    parser.add_argument('--print-freq', default=250, type=int)
    parser.add_argument('--exp-decay-rate', default=0.999, type=float)
    
    parser.add_argument('--word-dim', default=100, type=int)
    parser.add_argument('--n-head', default=4, type=int)
    parser.add_argument('--d-model', default=96, type=int)
    
    parser.add_argument('--DEC-LAYERS', default=3, type=int)
    parser.add_argument('--DEC-HEADS', default=4, type=int)

    parser.add_argument('--max-len-context', default=300, type=int)
    parser.add_argument('--max-len-answer', default=30, type=int)
    parser.add_argument('--max-len-question', default=30, type=int)
    
    parser.add_argument('--conv-num', default=4, type=int)
    parser.add_argument('--kernel-size', default=5, type=int)
    parser.add_argument('--CLIP', default=1, type=int)
    args = parser.parse_args()

    print('loading SQuAD data...')
    data = SQuAD(args)
    setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
    setattr(args, 'word_vocab_size', len(data.WORD.vocab))
    setattr(args, 'pad_idx_encoder', data.WORD.vocab.stoi[data.WORD.pad_token])
    setattr(args, 'pad_idx_decoder', data.WORD_DECODER.vocab.stoi[data.WORD_DECODER.pad_token])
    setattr(args, 'output_dim', len(data.WORD_DECODER.vocab))
    setattr(args, 'dataset_file', '.data/squad/{}'.format(args.dev_file))
    setattr(args, 'prediction_file', 'prediction{}.out'.format(args.gpu))
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
    setattr(args, 'device', torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"))
    print('data loading complete!')

    print('training start!')
    best_model = train(args, data)
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    if best_model:
        torch.save(best_model.state_dict(), 'saved_models/BiDAF_{}.pt'.format(args.model_time))
    print('training finished!')


if __name__ == '__main__':
    main()
