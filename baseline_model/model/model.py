import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Linear(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(Linear, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        self.reset_params()

    def reset_params(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.linear(x)
        return x

class Embedding(nn.Module):
    def __init__(self, args, pretrained):
        super().__init__()
        # char_emb
        # char_dim = 8, char_channel_width = 5
        # char_channel_size = 100
        # result: for each channel, we have 22 element -----> needs to be max-ppoled
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        
        self.char_conv = nn.Sequential(nn.Conv2d(1, args.char_channel_size, (args.char_dim, args.char_channel_width)), nn.ReLU())
        self.dropout = nn.Dropout(p=args.dropout)
        
        # word_emb
        # pre_trained GLoVe
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)
        assert self.args.hidden_size * 2 == (self.args.char_channel_size + self.args.word_dim)
        
        # high way network
        self.g = nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2), nn.ReLU())
        self.t = nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2), nn.ReLU())
    
    def forward(self, batch):
        char_embedding = self.char_emb_layer(batch.c_char)
        word_embedding = self.word_emb(batch.c_word[0])
        c = self.highway_network(char_embedding, word_embedding)
        return c
        
        
    def char_emb_layer(self, x):
        # x: (batch, seq_len, word_len)
        batch_size = x.size(0)
        
        embed = self.char_emb(x) # (batch, seq_len, word_len, char_dim)
        x = dropout(embed)
        
        # (batch * seq_len, 1, char_dim, word_len)
        x = x.view(-1, self.args.char_dim, x.size(2)).unsqueeze(1)
        
        # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
        x = self.char_conv(x).squeeze()
        
        # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
        x = F.max_pool1d(x, x.size(2)).squeeze()
        
        # (batch, seq_len, char_channel_size)
        x = x.view(batch_size, -1, self.args.char_channel_size)
        return x
    
    def highway_network(self, char, word):
        x = torch.cat([char, word], dim=-1)
        g = self.g(x)
        t = self.t(x)
        x = t * g + (1 - t) * x
        return x

class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        #self.n_head = 4
        #self.kqv_dim = 96
        #self.hidden_size = 100

        self.Wqs = [nn.Linear(2 * args.hidden_size, args.kqv_dim) for _ in range(args.n_head)]
        self.Wks = [nn.Linear(2 * args.hidden_size, args.kqv_dim) for _ in range(args.n_head)]
        self.Wvs = [nn.Linear(2 * args.hidden_size, args.kqv_dim) for _ in range(args.n_head)]
        
        
        for i in range(args.n_head):
            nn.init.xavier_uniform_(self.Wqs[i].weight)
            nn.init.xavier_uniform_(self.Wks[i].weight)
            nn.init.xavier_uniform_(self.Wvs[i].weight)

        self.attention_head_projection = nn.Linear(args.kqv_dim * args.n_head, 2 * args.hidden_size)
        nn.init.kaiming_uniform_(self.attention_head_projection.weight)
        
        self.norm_mh = nn.LayerNorm(2 * args.hidden_size)
        
    def forward(self, x):
        print(x.size())
        WQs, WKs, WVs = [], [], []
        
        for i in range(args.n_head):

            WQs.append(self.Wqs[i](x))
            WKs.append(self.Wks[i](x))
            WVs.append(self.Wvs[i](x))
        
        heads = []
        for i in range(args.n_head):
            out = torch.bmm(WQs[i], WKs[i].transpose(1, 2))
            out = torch.mul(out, 1 / math.sqrt(args.kqv_dim))
            out = F.softmax(out, dim=2)
            head_i = torch.bmm(out, WVs[i])
            heads.append(head_i)
        head = torch.cat(heads, dim=2)

        out = self.attention_head_projection(head)
        out = self.norm_mh(torch.add(out, x))
        
        return out
    
class EncoderBlock(nn.Module):
    
class Decoder(nn.Module):

class Baseline(nn.Module):
    def __init__(self, args, pretrained):
        super(Baseline, self).__init__()
        self.args = args
        
        # embedding layer
        self.emb = Embedding(args, pretrained)    
        
        # multihead self attention
        self.encoder1 = EncoderBlock()
        self.encdoer2 = EncoderBlock()
        
        # query and context attention
        self.CQAttention()
        
        # decoder
        self.Decoder = Decoder()
    
    def forward(self, batch, start, end):
        embedding = self.emb(batch)
        
        
#self.pos_emb = nn.Embedding()