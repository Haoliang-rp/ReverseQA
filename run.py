# -*- coding: utf-8 -*-
import argparse
import os

import torch
from torch import nn
from tensorboardX import SummaryWriter
from time import gmtime, strftime

from model.model import Baseline, Baseline_Bert
from model.data import SQuAD
#import evaluate
from torchtext.data.metrics import bleu_score
from tqdm import tqdm
import time
import copy

from transformers import BertModel, BertTokenizer, BertForQuestionAnswering

def train(args, data):
    QA_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad').to(args.device)
    if args.encoder_type == 'bert':
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(args.device) # AlbertModel.from_pretrained('albert-base-v2').to(args.device)
        model = Baseline_Bert(args, bert_model).to(args.device)
        if args.from_prev:
            print('loading {} model'.format(args.cur_model_path))
            model.load_state_dict(torch.load(args.cur_model_path))
    else:
        model = Baseline(args, data.WORD.vocab.vectors).to(args.device)

#    ema = EMA(args.exp_decay_rate)

    criterion = nn.CrossEntropyLoss(ignore_index = args.pad_idx_decoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate)

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    loss, last_epoch = 0, -1
    best_dev_loss = 20000
    best_train_loss = 20000
    ema = EMA(args.exp_decay_rate)

#    test_bound = 2
#    iterator = data.train_iter
    model.train()
    if args.encoder_type == 'bert': bert_model.eval()

    print('training')
    if args.fine_tune_bert: print('fine tune bert')
    for i, batch in enumerate(tqdm(data.train_iter)):
        start_time = time.time()

        present_epoch = int(data.train_iter.epoch)
        if present_epoch == args.epoch:
            break
#
        if present_epoch > last_epoch:
            print('epoch:', present_epoch + 1)
            if present_epoch > 0:
                if args.encoder_type == 'bert':
                    bleu_score, overlapping_score = calculate_bleu_bert(args, data.dev, bert_model, model, QA_model)
                    print('bleu score after {} epoch is {}'. format(last_epoch, bleu_score))
                    print('overlapping score after {} epoch is {}'. format(last_epoch, overlapping_score))
                else:
                    bleu_score = calculate_bleu(data.dev, model, args.device)
                    print('bleu score after {} epoch is {}'. format(last_epoch, bleu_score))

        last_epoch = present_epoch

        optimizer.zero_grad()

        if args.encoder_type == 'bert':
            training_batch = list(zip(batch.answer, batch.context))
            training_batch_in = args.tokenizer.batch_encode_plus(training_batch, add_special_tokens=True, pad_to_max_length=True, return_tensors="pt")

            question_batch_in = args.decoder_tokenizer.batch_encode_plus(batch.question, add_special_tokens=True, pad_to_max_length=True, return_tensors="pt")

            if args.fine_tune_bert:
                input_ids_tensor = training_batch_in['input_ids'].to(args.device)
                attention_mask_tensor = training_batch_in['attention_mask'].to(args.device)
                token_type_ids_tensor = training_batch_in['token_type_ids'].to(args.device)

                cmask = copy.deepcopy(token_type_ids_tensor).unsqueeze(1).unsqueeze(2)#.unsqueeze(2).repeat(1, 1, 768)

                token_type_ids_tensor = add_pos_info(token_type_ids_tensor, batch.s_idx, batch.e_idx, batch.context, args.tokenizer)

                if input_ids_tensor.size(1) > 511: continue
                outputs = bert_model(input_ids_tensor, attention_mask_tensor, token_type_ids_tensor)
                encoded = outputs[0]

                question_input_ids = question_batch_in['input_ids'].to(args.device)
                if question_input_ids.size(1) > args.max_len_question + 9: continue
            else:
                with torch.no_grad():
                    input_ids_tensor = training_batch_in['input_ids'].to(args.device)
                    attention_mask_tensor = training_batch_in['attention_mask'].to(args.device)
                    token_type_ids_tensor = training_batch_in['token_type_ids'].to(args.device)

                    cmask = copy.deepcopy(token_type_ids_tensor).unsqueeze(1).unsqueeze(2)#.unsqueeze(2).repeat(1, 1, 768)

                    token_type_ids_tensor = add_pos_info(token_type_ids_tensor, batch.s_idx, batch.e_idx, batch.context, args.tokenizer)

                    if input_ids_tensor.size(1) > 511: continue
                    outputs = bert_model(input_ids_tensor, attention_mask_tensor, token_type_ids_tensor)
                    encoded = outputs[0]

                    question_input_ids = question_batch_in['input_ids'].to(args.device)
                    if question_input_ids.size(1) > args.max_len_question + 9: continue

            mask = attention_mask_tensor.unsqueeze(1).unsqueeze(2)
            X, _ = model(encoded, mask, question_input_ids, cmask)

            output_dim = X.shape[-1]
            X = X.contiguous().view(-1, output_dim)

            # question_batch_in['input_ids']
            label = question_input_ids[:,1:].contiguous().view(-1).to(args.device)

        else:
            context_word, context_char = batch.c_word[0], batch.c_char
            answer_word, answer_char = batch.a_word[0], batch.a_char
            question_word, question_char = batch.q_word_decoder[0][:,:-1], batch.q_char_decoder[:,:-1]

            X, _ = model(context_word, context_char, answer_word, answer_char, question_word, question_char)
            output_dim = X.shape[-1]
            X = X.contiguous().view(-1, output_dim)

            label = batch.q_word_decoder[0][:, 1:].contiguous().view(-1)

        batch_loss = criterion(X, label)
        loss += batch_loss.item()
        batch_loss.backward()

        optimizer.step()

        for name, p in model.named_parameters():
            if p.requires_grad: ema.update_parameter(name, p)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.CLIP)

        ques_beam = generate_question_bert_enc_beam_search(args, batch.answer[0], batch.context[0], batch.s_idx[0], batch.e_idx[0], bert_model, model, args.beam_size)
        print(ques_beam)
        case = 0
        if (i + 1) % args.print_freq == 0:
            if args.encoder_type == 'bert':
                dev_loss = test(args, model, data, bert_model)#, ema
            else:
                dev_loss = test(args, model, data)
            best_dev_loss = min(dev_loss, best_dev_loss)

            best_train_loss = min(batch_loss, best_train_loss)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print('Time: {}m {}s'.format(epoch_mins, epoch_secs))
            print('train loss: {} | dev loss: {}'.format(batch_loss, dev_loss))

            if args.encoder_type == 'bert':
                ques, att = generate_question_bert_enc(args, batch.answer[0], batch.context[0], batch.s_idx[0], batch.e_idx[0], bert_model, model)
                ques_beam = generate_question_bert_enc_beam_search(args, batch.answer[0], batch.context[0], batch.s_idx[0], batch.e_idx[0], bert_model, model, args.beam_size)
                print('sample question: {}'.format(' '.join(ques)))
                print('sample question beam search: ')
                print(ques_beam)
                print('real question: {}'.format(batch.question[0]))
            else:

                c_word, c_char, a_word, a_char = data.dev.examples[case].c_word, data.dev.examples[case].c_char, data.dev.examples[case].a_word, data.dev.examples[case].a_char#
                ques, att = generate_question(args, c_word, c_char, a_word, a_char, model, data)
                print('sample question: {}'.format(' '.join(ques)))
                print('real question: {}'.format(' '.join(data.train.examples[case].q_word)))
                case += 1

            c = (i + 1) // args.print_freq

            writer.add_scalar('loss/train', batch_loss, c)
            writer.add_scalar('loss/dev', dev_loss, c)

            if (i + 1) % args.save_freq == 0 and dev_loss <= best_dev_loss:
                print('saving model(dev)')
#                torch.save(bert_model.state_dict(), 'saved_bert_models/BASE_{}_{}.pt'.format(args.encoder_type, args.model_time))
                torch.save(model.state_dict(), 'saved_models/BASE_{}_{}.pt'.format(args.encoder_type, args.model_time))

            if (i + 1) % args.save_freq == 0 and batch_loss <= best_train_loss:
                print('saving model(train)')
#                torch.save(bert_model.state_dict(), 'saved_bert_models/BASE_Train_{}_{}.pt'.format(args.encoder_type, args.model_time))
                torch.save(model.state_dict(), 'saved_models/BASE_Train_{}_{}.pt'.format(args.encoder_type, args.model_time))

    return model

def test(args, model, data, bert_model=None):#, ema
    criterion = nn.CrossEntropyLoss(ignore_index = args.pad_idx_decoder)
    loss = 0

    model.eval()
    if bert_model: bert_model.eval()

    print('testing')
    with torch.set_grad_enabled(False):
        num = 0
        for batch in iter(tqdm(data.dev_iter)):
            num += 1
            if num > 20: break
            if args.encoder_type == 'bert':
                training_batch = list(zip(batch.answer, batch.context))
                training_batch_in = args.tokenizer.batch_encode_plus(training_batch, add_special_tokens=True, pad_to_max_length=True, return_tensors="pt")

                question_batch_in = args.decoder_tokenizer.batch_encode_plus(batch.question, add_special_tokens=True, pad_to_max_length=True, return_tensors="pt")

#                with torch.no_grad():
                input_ids_tensor = training_batch_in['input_ids'].to(args.device)
                attention_mask_tensor = training_batch_in['attention_mask'].to(args.device)
                token_type_ids_tensor = training_batch_in['token_type_ids'].to(args.device)
                outputs = bert_model(input_ids_tensor, attention_mask_tensor, token_type_ids_tensor)
                encoded = outputs[0]

                cmask = copy.deepcopy(token_type_ids_tensor).unsqueeze(1).unsqueeze(2)

                token_type_ids_tensor = add_pos_info(token_type_ids_tensor, batch.s_idx, batch.e_idx, batch.context, args.tokenizer)

                question_input_ids = question_batch_in['input_ids'].to(args.device)

                mask = attention_mask_tensor.unsqueeze(1).unsqueeze(2)
                cmask = token_type_ids_tensor.unsqueeze(1).unsqueeze(2)#.unsqueeze(2).repeat(1, 1, 768)
                X, _ = model(encoded, mask, question_input_ids, cmask)

                output_dim = X.shape[-1]
                X = X.contiguous().view(-1, output_dim)

                label = question_batch_in['input_ids'][:,1:].contiguous().view(-1).to(args.device)

            else:
                context_word, context_char = batch.c_word[0], batch.c_char
                answer_word, answer_char = batch.a_word[0], batch.a_char
                question_word, question_char = batch.q_word_decoder[0][:,:-1], batch.q_char_decoder[:,:-1]

                X, _ = model(context_word, context_char, answer_word, answer_char, question_word, question_char)
                output_dim = X.shape[-1]
                X = X.contiguous().view(-1, output_dim)

                label = batch.q_word_decoder[0][:, 1:].contiguous().view(-1)

            batch_loss = criterion(X, label)
            loss += batch_loss.item()


    return loss / num

def add_pos_info(token_type_ids_tensor, s_idx, e_idx, context, tokenizer):
    for i in range(len(context)):
        context_token_len = len(tokenizer.tokenize(context[i]))

        token_type_ids_tensor[i][-context_token_len+s_idx[i]:-context_token_len+e_idx[i]+1] = \
        torch.zeros_like(token_type_ids_tensor[i][-context_token_len+s_idx[i]:-context_token_len+e_idx[i]+1])
    return token_type_ids_tensor

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def calculate_bleu(data, model, device, max_len = 30):
    labels = []
    preds = []

    print('calculating bleu score')
    for i in tqdm(range(500)):
        datum = data.examples[i]
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
        labels.append([datum.q_word])

    return bleu_score(preds, labels, max_n=2, weights=[0.5, 0.5])

def calculate_bleu_bert(args, data, bert_model, model, QA_model):
    labels = []
    preds = []

    print('calculating bleu score for 500 questions')
    num = 0
    for i in tqdm(range(500)):
        example = data.examples[i]
        answer = example.answer
        context = example.context
        question = example.question
        s_idx_true = example.s_idx
        e_idx_true = example.e_idx
#        ques_token = args.decoder_tokenizer.encode_plus(question, add_special_tokens=False, pad_to_max_length=False, return_tensors="pt")
#        ques_token = args.decoder_tokenizer.convert_ids_to_tokens(ques_token['input_ids'][0])
        ques_token = args.decoder_tokenizer.tokenize(question)

        pred, _ = generate_question_bert_enc(args, answer, context, s_idx_true, e_idx_true, bert_model, model)

        # calculatiing overlap score
        generated_question = ' '.join(pred)

        encoding = args.tokenizer.encode_plus(generated_question, context)
        input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
        start_scores, end_scores = QA_model(torch.tensor([input_ids]).to(args.device), token_type_ids=torch.tensor([token_type_ids]).to(args.device))

        tokenized_ques = args.tokenizer.tokenize(generated_question)
        cur_question_len = len(tokenized_ques)
        s_idx = torch.argmax(start_scores) - cur_question_len - 2
        e_idx = torch.argmax(end_scores) - cur_question_len - 2

        score = calculate_overlapping_score(s_idx_true, e_idx_true+1, s_idx.item(), e_idx.item()+1)
        num += 1

        preds.append(pred)
        labels.append([ques_token])

    return bleu_score(preds, labels, max_n=2, weights=[0.5, 0.5]), score / num

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

    context_word, context_char = WORD.process(c_word)[1].unsqueeze(0).to(device), CHAR.process(c_char).squeeze(2).unsqueeze(0).to(device)
    answer_word, answer_char = WORD.process(a_word)[1].unsqueeze(0).to(device), CHAR.process(a_char).squeeze(2).unsqueeze(0).to(device)

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

def generate_question_bert_enc(args, answer, context, s_idx, e_idx, bert_model, model, max_len_question=30):
    '''
    answer: untokenized string
    context: untokenized string
    '''
    device = args.device
    bert_model.eval()
    model.eval()
    test_pair_in = args.tokenizer.encode_plus(answer, context, add_special_tokens=True, pad_to_max_length=True, return_tensors="pt")

    with torch.no_grad():
        input_ids_tensor = test_pair_in['input_ids'].to(device)
        attention_mask_tensor = test_pair_in['attention_mask'].to(device)
        token_type_ids_tensor = test_pair_in['token_type_ids'].to(device)

        context_token_len = len(args.tokenizer.tokenize(context))

        cmask = copy.deepcopy(token_type_ids_tensor).unsqueeze(1).unsqueeze(2)

        token_type_ids_tensor[0][-context_token_len+s_idx:-context_token_len+e_idx+1] = \
        torch.zeros_like(token_type_ids_tensor[0][-context_token_len+s_idx:-context_token_len+e_idx+1])

        outputs = bert_model(input_ids_tensor, attention_mask_tensor, token_type_ids_tensor)
        encoded = outputs[0]

        attentions = []

        word_idxes = [args.decoder_tokenizer.cls_token_id]
        for i in range(max_len_question+2):
            word_tensor = torch.LongTensor(word_idxes).unsqueeze(0).to(device)
            word_mask = model.make_dec_mask(word_tensor)

            with torch.no_grad():
                Q_emb = model.emb(word_tensor)
                output, attention = model.decoder(Q_emb, encoded, word_mask, cmask)

            attentions.append(attention)

            pred_token = output.argmax(2)[:,-1].item()
            word_idxes.append(pred_token)
#            question_input_ids[0][i+1] = pred_token
#            question_attention_mask[0][i+1] = 1

            if pred_token == args.decoder_tokenizer.sep_token_id: break

        qus = args.decoder_tokenizer.convert_ids_to_tokens(word_idxes)
        return qus[1:i+1], attentions


def generate_question_bert_enc_beam_search(args, answer, context, s_idx, e_idx, bert_model, model, k, max_len_question=33):
    '''
    answer: untokenized string
    context: untokenized string
    k: beam size
    '''
    device = args.device
    bert_model.eval()
    model.eval()
    test_pair_in = args.tokenizer.encode_plus(answer, context, add_special_tokens=True, pad_to_max_length=True, return_tensors="pt")

    with torch.no_grad():
        input_ids_tensor = test_pair_in['input_ids'].to(device)
        attention_mask_tensor = test_pair_in['attention_mask'].to(device)
        token_type_ids_tensor = test_pair_in['token_type_ids'].to(device)

        context_token_len = len(args.tokenizer.tokenize(context))

        cmask = copy.deepcopy(token_type_ids_tensor).unsqueeze(1).unsqueeze(2)

        token_type_ids_tensor[0][-context_token_len+s_idx:-context_token_len+e_idx+1] = \
        torch.zeros_like(token_type_ids_tensor[0][-context_token_len+s_idx:-context_token_len+e_idx+1])

        outputs = bert_model(input_ids_tensor, attention_mask_tensor, token_type_ids_tensor)
        encoded = outputs[0]

        word_idxes = [([args.decoder_tokenizer.cls_token_id], 1)]

        ques = []
        for i in range(max_len_question+2):
            word_tensor = [word_idx for word_idx, prob in word_idxes]

            word_tensor = torch.LongTensor(word_tensor).to(device)
            word_mask = model.make_dec_mask(word_tensor)
#             print(word_tensor.size())
            Q_emb = model.emb(word_tensor)
#             print(Q_emb.size())
            encoded = outputs[0].repeat(word_tensor.size(0), 1, 1)
#             print(encoded.size())
            with torch.no_grad():
                out, attention = model.decoder(Q_emb, encoded, word_mask, cmask)

            out = F.softmax(out, dim=2)
            candiate = []
            for j in range(len(word_tensor)):
                idx = word_idxes[j][0]
                prev_prob = word_idxes[j][1]

                for q in range(out.size(2)):
                    candiate.append((idx+[q], prev_prob * -math.log(out[j,-1,q].item())))
            candiate.sort(key=lambda x:x[1])
            word_idxes = candiate[:k]

            dummy = copy.deepcopy(word_idxes)
            for item in dummy:
                word_idx, prob = item
                if word_idx[-1] == args.decoder_tokenizer.sep_token_id:
                    ques.append(word_idx)
                    word_idxes.remove(item)

            if not word_idxes or len(ques) > k+2: break

        if len(ques) < k:
            for word_idx, prob in word_idxes:
                ques.append(word_idx)

        print(ques)
        res = []
        for que in ques:
            if que[-1] == args.decoder_tokenizer.sep_token_id:
                word = args.decoder_tokenizer.convert_ids_to_tokens(que)[1:-1]
                res.append(' '.join(word))
            else:
                word = args.decoder_tokenizer.convert_ids_to_tokens(que)[1:]
                res.append(' '.join(word))
        return res

def calculate_overlapping_score(s_idx_true, e_idx_true, s_idx_g, e_idx_g):
    x = range(s_idx_true, e_idx_true)
    y = range(s_idx_g, e_idx_g)

    xs = set(x)

    return len(xs.intersection(y)) / len(xs.union(y))


class EMA(object):
    def __init__(self, decay):
        self.decay = decay
        self.shadows = {}
        self.devices = {}

    def __len__(self):
        return len(self.shadows)

    def get(self, name: str):
        return self.shadows[name].to(self.devices[name])

    def set(self, name: str, param: nn.Parameter):
        self.shadows[name] = param.data.to('cpu').clone()
        self.devices[name] = param.data.device

    def update_parameter(self, name: str, param: nn.Parameter):
        if name in self.shadows:
            data = param.data
            new_shadow = self.decay * data + (1.0 - self.decay) * self.get(name)
            param.data.copy_(new_shadow)
            self.shadows[name] = new_shadow.to('cpu').clone()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder-type', default='bert')

    parser.add_argument('--dev-batch-size', default=30, type=int)
    parser.add_argument('--dev-file', default='dev-v2.0.json')
    parser.add_argument('--train-batch-size', default=30, type=int)
    parser.add_argument('--train-file', default='train-v2.0.json')

    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--learning-rate', default=0.0005, type=float)
    parser.add_argument('--exp-decay-rate', default=0.999, type=float)

    parser.add_argument('--word-dim', default=100, type=int)
    parser.add_argument('--n-head', default=4, type=int)

    parser.add_argument('--DEC-LAYERS', default=3, type=int)
#    parser.add_argument('--DEC-HEADS', default=4, type=int)

    parser.add_argument('--max-len-context', default=300, type=int)
    parser.add_argument('--max-len-answer', default=30, type=int)
    parser.add_argument('--max-len-question', default=30, type=int)
    parser.add_argument('--max-len-char', default=30, type=int)

    parser.add_argument('--kernel-size', default=5, type=int)
    parser.add_argument('--CLIP', default=1, type=int)

    parser.add_argument('--print-freq', default=100, type=int)
    parser.add_argument('--save-freq', default=100, type=int)
    parser.add_argument('--epoch', default=18, type=int)
#    parser.add_argument('--decaying-rate', default=0.98, type=int)

    parser.add_argument('--cur-model-path', default='saved_models/BASE_bert_09:22:02.pt')
    parser.add_argument('--from-prev', default=False)
    parser.add_argument('--fine-tune-bert', default=True)
    parser.add_argument('--beam-size', default=3, type=int)

    args = parser.parse_args()
    setattr(args, 'device', torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"))#'cpu'
    print('loading SQuAD data...')

    data = SQuAD(args)
    if args.encoder_type == 'bert':
        setattr(args, 'd_model', 768 // args.n_head)
        setattr(args, 'tokenizer', BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, padding_side='left'))
        setattr(args, 'decoder_tokenizer', BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, padding_side='right'))
        setattr(args, 'pad_idx_decoder', args.decoder_tokenizer.pad_token_id)
        setattr(args, 'output_dim', args.tokenizer.vocab_size)
        setattr(args, 'hidden_size', 768 // 2) # bert have size 128 for word embedding
    else:
        setattr(args, 'd_model', 96) # 768 // args.n_head
        setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
        setattr(args, 'word_vocab_size', len(data.WORD.vocab))
        setattr(args, 'pad_idx_encoder', data.WORD.vocab.stoi[data.WORD.pad_token])
        setattr(args, 'pad_idx_decoder', data.WORD_DECODER.vocab.stoi[data.WORD_DECODER.pad_token])
        setattr(args, 'output_dim', len(data.WORD_DECODER.vocab))
        setattr(args, 'hidden_size', 100)
        setattr(args, 'char_dim', 8)
        setattr(args, 'char_channel_width', 3)
        setattr(args, 'char_channel_size', 100)
        setattr(args, 'conv_num', 8)
#        setattr(args, 'dataset_file', '.data/squad/{}'.format(args.dev_file))

    setattr(args, 'prediction_file', 'prediction{}.out'.format(args.gpu))
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))

    print('data loading complete!')

    print('training start!')
    best_model = train(args, data)
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    if best_model:
        torch.save(best_model.state_dict(), 'saved_models/Baseline_{}_{}.pt'.format(args.encoder_type, args.model_time))
    print('training finished!')


if __name__ == '__main__':
    main()
