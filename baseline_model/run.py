# -*- coding: utf-8 -*-
import argparse
import copy, json, os

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from time import gmtime, strftime

from model.model import Baseline
from model.data import SQuAD, word_tokenize
from model.ema import EMA
#import evaluate
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
import math
import time

def train(args, data):
    model = Baseline(args, data.WORD.vocab.vectors).to(args.device)
    
#    ema = EMA(args.exp_decay_rate)
#    for name, param in model.named_parameters():
#        if param.requires_grad:
#            ema.update(name, param.data)
#    parameters = filter(lambda p:p.requires_grad, model.parameters())

    criterion = nn.CrossEntropyLoss(ignore_index = args.pad_idx_decoder)
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
        start_time = time.time()
        
        present_epoch = int(data.train_iter.epoch)
        if present_epoch == args.epoch:
            break
#        
        if present_epoch > last_epoch:
            print('epoch:', present_epoch + 1)
        last_epoch = present_epoch
        
        optimizer.zero_grad()
        
        context_word, context_char = batch.c_word[0], batch.c_char
        answer_word, answer_char = batch.a_word[0], batch.a_char
        question_word, question_char = batch.q_word_decoder[0][:,:-1], batch.q_char_decoder[:,:-1]
        
        X, _ = model(context_word, context_char, answer_word, answer_char, question_word, question_char)
        output_dim = X.shape[-1]
        X = X.contiguous().view(-1, output_dim)

        label = batch.q_word_decoder[0][:, 1:].contiguous().view(-1)
#        
        
        batch_loss = criterion(X, label)
        loss += batch_loss.item()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.CLIP)
        optimizer.step()
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
#        for name, param in model.named_parameters():
#            if param.requires_grad:
#                ema.update(name, param.data)
#        
        if (i + 1) % args.print_freq == 0:
            print('Time: {}m {}s'.format(epoch_mins, epoch_secs))
            dev_loss = test(args, model, data)#, ema
            print('train loss: {} | dev loss: {}'.format(batch_loss, dev_loss))
            print('tran loss PPL: {} | dev loss PPL: {}'.format(math.exp(batch_loss), math.exp(dev_loss)))
            
        if (i + 1) % args.save_freq == 0:
            print('saving model')
            torch.save(model.state_dict(), 'saved_models/BiDAF_{}.pt'.format(args.model_time))
 
    return model

def test(args, model, data):#, ema
    criterion = nn.CrossEntropyLoss(ignore_index = args.pad_idx_decoder)
    loss = 0
    
    model.eval()
    
#    backup_params = EMA(0)
#    for name, param in model.named_parameters():
#        if param.requires_grad:
#            backup_params.register(name, param.data)
#            param.data.copy_(ema.get(name))
            
    with torch.set_grad_enabled(False):
        for batch in iter(data.dev_iter):
            context_word, context_char = batch.c_word[0], batch.c_char
            answer_word, answer_char = batch.a_word[0], batch.a_char
            question_word, question_char = batch.q_word_decoder[0][:,:-1], batch.q_char_decoder[:,:-1]
            
            X, _ = model(context_word, context_char, answer_word, answer_char, question_word, question_char)
            output_dim = X.shape[-1]
            X = X.contiguous().view(-1, output_dim)
            
            label = batch.q_word_decoder[0][:, 1:].contiguous().view(-1)
            
            batch_loss = criterion(X, label)
            loss += batch_loss.item()
            
#        for name, param in model.named_parameters():
#            if param.requires_grad:
#                param.data.copy_(backup_params.get(name))

    return loss

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def calculate_bleu(data, model, device, max_len = 30):
    labels = []
    preds = []
    
    print('calculating bleu score')
    for datum in tqdm(data.examples):
        
        c_word = datum.c_word
        c_char = datum.c_char
        a_word = datum.a_word
        a_char = datum.a_char
        
        try:
            pred, _ = generate_question(c_word, c_char, a_word, a_char, model, device)
        except:
            continue
        
        pred = pred[:-1]
        
        preds.append(pred)
        labels.append(datum.q_word)
    
    return bleu_score(preds, labels)
                
def generate_question(args, c_word, c_char, a_word, a_char, model, data):
    '''
    c_word: list: tokenized context (for example: form data.examples[0].c_word)
    c_char: 2d list: tokenized char
    a_word: list: tokenized context
    a_char: 2d list: tokenized char
    '''
    device = args.device
    max_len_question = args.max_len_question
    max_len_char = args.max_len_char
    
    WORD = data.WORD
    CHAR = data.CHAR
    WORD_DECODER = data.WORD_DECODER
    CHAR_DECODER = data.CHAR_DECODER
    
    model.eval()
    
    context_word, context_char = WORD.process(c_word)[1].unsqueeze(0), CHAR.process(c_char).squeeze(2).unsqueeze(0)
    answer_word, answer_char = WORD.process(a_word)[1].unsqueeze(0), CHAR.process(a_char).squeeze(2).unsqueeze(0)
    
    cmask = model.make_enc_mask(context_word).to(device)
    amask = model.make_enc_mask(answer_word).to(device)
    
    with torch.no_grad():
        C_emb = model.emb(context_char, context_word)
        A_emb = model.emb(answer_char, answer_word)
        
        C = model.context_conv(C_emb.permute(0, 2, 1))
        A = model.answer_conv(A_emb.permute(0, 2, 1))
        
        Ce = model.c_enc(C, cmask)
        Ae = model.a_enc(A, amask)
        
        encoded = model.ca_att(Ce, Ae, cmask, amask).to(device)
    
    word_idxes = [WORD_DECODER.vocab.stoi[WORD_DECODER.init_token]]
    word = []
    char_tensor_list = [[CHAR_DECODER.vocab.stoi[CHAR_DECODER.init_token]] + [CHAR_DECODER.vocab.stoi[CHAR_DECODER.pad_token]] * (max_len_char - 1) ] 
    
    for i in range(max_len_question+2):
        word_tensor = torch.LongTensor(word_idxes).unsqueeze(0).to(device)
        word_mask = model.make_dec_mask(word_tensor)
        
        char_tensor = torch.LongTensor(char_tensor_list).unsqueeze(0).to(device)
        Q_emb = model.emb(char_tensor, word_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(Q_emb, encoded, word_mask, cmask)
        
        pred_token = output.argmax(2)[:,-1].item()
            
        word_idxes.append(pred_token)
        word.append(WORD_DECODER.vocab.itos[pred_token])

        char_enco_list = CHAR_DECODER.process(list(word[-1]))[:,1:-1,:].squeeze().tolist()
        char_enco_list = [char_enco_list] if isinstance(char_enco_list, int) else char_enco_list
        pad_len = max_len_char - len(char_enco_list)
        char_enco_list = char_enco_list + [CHAR_DECODER.vocab.stoi[CHAR_DECODER.pad_token]] * pad_len
        char_tensor_list.append(char_enco_list)
        
        if pred_token == WORD_DECODER.vocab.stoi[WORD_DECODER.eos_token]:
            break
    return word, attention
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-dim', default=8, type=int)
    parser.add_argument('--char-channel-width', default=3, type=int)
    parser.add_argument('--char-channel-size', default=100, type=int)
    
    parser.add_argument('--dev-batch-size', default=100, type=int)
    parser.add_argument('--dev-file', default='dev-v2.0.json')
    parser.add_argument('--train-batch-size', default=60, type=int)
    parser.add_argument('--train-file', default='train-v2.0.json')
    
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    
    parser.add_argument('--learning-rate', default=0.05, type=float)
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
    
    parser.add_argument('--print-freq', default=250, type=int)
    parser.add_argument('--save-freq', default=500, type=int)
    parser.add_argument('--epoch', default=12, type=int)
    
    args = parser.parse_args()
    setattr(args, 'device', torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"))#
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
