# -*- coding: utf-8 -*-
import json
import os
import nltk
import torch
import pickle
from tqdm import tqdm

from torchtext import data
from torchtext.vocab import GloVe

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

class SQuAD():
    def __init__(self, args):
        path = 'data'
        dataset_path = path + '/torchtext/'
        train_examples_path = dataset_path + 'train_examples.pt'
        dev_examples_path = dataset_path + 'dev_examples.pt'
        
        self.max_len_context = args.max_len_context
        self.max_len_question = args.max_len_question
        self.max_len_answer = args.max_len_answer

        print("preprocessing data files...")
        if not os.path.exists('{}/{}l'.format(path, args.train_file)):
            self.preprocess_file('{}/{}'.format(path, args.train_file), create_question=True)
        if not os.path.exists('{}/{}l'.format(path, args.dev_file)):
            self.preprocess_file('{}/{}'.format(path, args.dev_file))

        self.RAW = data.RawField()

        self.RAW.is_target = False
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
        
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=word_tokenize)
        self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True)
        
        self.CHAR_NESTING_DECODER = data.Field(batch_first=True, tokenize=list, lower=True)
        self.CHAR_DECODER = data.NestedField(self.CHAR_NESTING_DECODER, tokenize=word_tokenize, init_token = "<sos>", eos_token = "<eos>")
        self.WORD_DECODER = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True, init_token = "<sos>", eos_token = "<eos>")
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {
            'id': ('id', self.RAW),
            's_idx': ('s_idx', self.LABEL),
            'e_idx': ('e_idx', self.LABEL),
            'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
            'question': [('q_word', self.WORD), ('q_char', self.CHAR), ('q_word_decoder', self.WORD_DECODER), ('q_char_decoder', self.CHAR_DECODER)],
            'answer': [('a_word', self.WORD), ('a_char', self.CHAR)]
        }

        list_fields = [
            ('id', self.RAW), 
            ('s_idx', self.LABEL), ('e_idx', self.LABEL),
            ('c_word', self.WORD), ('c_char', self.CHAR),
            ('q_word', self.WORD), ('q_char', self.CHAR),
            ('q_word_decoder', self.WORD_DECODER), ('q_char_decoder', self.CHAR_DECODER),
            ('a_word', self.WORD), ('a_char', self.CHAR)
        ]
        
        if os.path.exists(dataset_path):
            print("loading splits...")
            train_examples = torch.load(train_examples_path)
            dev_examples = torch.load(dev_examples_path)

            self.train = data.Dataset(examples=train_examples, fields=list_fields)
            self.dev = data.Dataset(examples=dev_examples, fields=list_fields)
        else:
            print("building splits...")
            self.train, self.dev = data.TabularDataset.splits(
                path=path,
                train='{}l'.format(args.train_file),
                validation='{}l'.format(args.dev_file),
                format='json',
                fields=dict_fields)

            os.makedirs(dataset_path)
            torch.save(self.train.examples, train_examples_path)
            torch.save(self.dev.examples, dev_examples_path)

        print("building vocab...")
        self.CHAR.build_vocab(self.train, self.dev)
        self.CHAR_DECODER.build_vocab(self.train, self.dev)
        self.WORD.build_vocab(self.train, self.dev, vectors=GloVe(name='6B', dim=args.word_dim))
        self.WORD_DECODER.build_vocab(self.train, self.dev, vectors=GloVe(name='6B', dim=args.word_dim))

        print("building iterators...")
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        self.train_iter = data.BucketIterator(
            self.train,
            batch_size=args.train_batch_size,
            device=device,
            repeat=True,
            shuffle=True,
            sort_key=lambda x: len(x.c_word)
        )

        self.dev_iter = data.BucketIterator(
            self.dev,
            batch_size=args.dev_batch_size,
            device=device,
            repeat=False,
            sort_key=lambda x: len(x.c_word)
        )
        
    def preprocess_file(self, path, create_question=False):
        dump = []
        questions = []
        alignment_problems = 0
        num_impossible = 0
        abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']
    
        with open(path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            json_data = json_data['data']
    
            for article in tqdm(json_data):
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    
                    cur_context_len = len(word_tokenize(context))
                    if cur_context_len > self.max_len_context:
                        continue
                    
                    tokens = word_tokenize(context)
                    for qa in paragraph['qas']:
                        id = qa['id']
                        question = qa['question']
                        
                        cur_question_len = len(word_tokenize(question))
                        if cur_question_len > self.max_len_question:
                            continue
                        
                        if create_question:
                            questions.append(question)
                        
                        if qa['is_impossible']:
                            num_impossible += 1
                            continue
                            
                        for ans in qa['answers']:
                            answer = ans['text']
                            
                            cur_answer_len = len(word_tokenize(question))
                            if cur_answer_len > self.max_len_answer:
                                continue
                            
                            s_idx = ans['answer_start']
                            e_idx = s_idx + len(answer)
                            
                            
                            l = 0
                            s_found = False
                            for i, t in enumerate(tokens):
                                while l < len(context):
                                    if context[l] in abnormals:
                                        l += 1
                                    else:
                                        break
                                # exceptional cases
                                if t[0] == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\'' + t[1:]
                                elif t == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\''
    
                                l += len(t)
                                if l > s_idx and s_found == False:
                                    s_idx = i
                                    s_found = True
                                if l >= e_idx:
                                    e_idx = i
                                    break
                            
                            
                            if ''.join(tokens[s_idx: e_idx+1]) != answer.replace(" ", ""):
    #                             print(''.join(tokens[s_idx: e_idx+1]))
    #                             print(answer.replace(" ", ""))
                                alignment_problems += 1
                            
                            dump.append(dict([('id', id),
                                              ('context', context),
                                              ('question', question),
                                              ('answer', answer),
                                              ('s_idx', s_idx),
                                              ('e_idx', e_idx)]))
            
            with open('{}l'.format(path), 'w', encoding='utf-8') as f:
                for line in dump:
                    json.dump(line, f)
                    print('', file=f)
            
            if create_question:
                with open('questions.pickle', 'wb') as f:
                    pickle.dump(questions, f)
                    
            print('--data have alignment_problems: {}'.format(alignment_problems))
            print('--questions do not have answer: {}'.format(num_impossible))