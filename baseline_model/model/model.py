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
        self.args = args
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        
        self.char_conv = nn.Sequential(nn.Conv2d(1, args.char_channel_size, (args.char_dim, args.char_channel_width)), nn.ReLU())
        self.dropout = nn.Dropout(p=args.dropout)
        
        # word_emb
        # pre_trained GLoVe
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)
        assert args.hidden_size * 2 == (args.char_channel_size + args.word_dim)
        
        # high way network
        self.g = nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2), nn.ReLU())
        self.t = nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2), nn.ReLU())
    
    def forward(self, char, word):
        char_embedding = self.char_emb_layer(char)
        word_embedding = self.word_emb(word)
        c = self.highway_network(char_embedding, word_embedding)
        return c
        
        
    def char_emb_layer(self, x):
        # x: (batch, seq_len, word_len)
        batch_size = x.size(0)
        
        embed = self.char_emb(x) # (batch, seq_len, word_len, char_dim)
        x = self.dropout(embed)
        
        # (batch * seq_len, 1, char_dim, word_len)
        x = x.view(-1, self.args.char_dim, x.size(2)).unsqueeze(1)
        
        # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
        x = self.char_conv(x).squeeze(2)
        
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
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")
        
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)
    
    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_head, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_head = n_head
        self.head_dim = self.hid_dim // self.n_head
        
        self.q_linear = nn.Linear(self.hid_dim, self.hid_dim)
        self.k_linear = nn.Linear(self.hid_dim, self.hid_dim)
        self.v_linear = nn.Linear(self.hid_dim, self.hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(self.hid_dim, self.hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    
    def forward(self, query, key, value, mask=None):
        # batch_size, seq_len, hid_dim
        batch_size = query.size(0)
        
        # batch_size x n_head x seq_len x head_dim
        Q = self.q_linear(query).view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_linear(key).view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_linear(value).view(batch_size, -1, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        
        # batch_size x n_head x seq_len x head_dim   matmul   batch_size x n_head x head_dim x seq_len
        # ----> batch_size x n_head x seq_len x seq_len
        attention_matrix = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        
        if mask is not None:
            attention_matrix = attention_matrix.masked_fill(mask==0, -1e10)
            
        attention = torch.softmax(attention_matrix, dim=-1)
        
        # batch_size x n_head x seq_len x head_dim
        x = torch.matmul(self.dropout(attention), V)
        
        # makes a copy of tensor so the order of elements would be same as if tensor of same shape created from scratch
        # batch_size x seq_len x n_head x head_dim
        x = x.permute(0, 2, 1, 3).contiguous()
        
        # batch_size x seq_len x hid_dim
        x = x.view(batch_size, -1, self.hid_dim)
        
        # batch_size x seq_len x hid_dim
        x = self.fc(x)

        return x, attention
# =============================================================================
# class SelfAttention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         #self.n_head = 4
#         #self.kqv_dim = 96
#         #self.hidden_size = 100
# 
#         self.Wqs = [nn.Linear(2 * args.hidden_size, args.kqv_dim) for _ in range(args.n_head)]
#         self.Wks = [nn.Linear(2 * args.hidden_size, args.kqv_dim) for _ in range(args.n_head)]
#         self.Wvs = [nn.Linear(2 * args.hidden_size, args.kqv_dim) for _ in range(args.n_head)]
#         
#         
#         for i in range(args.n_head):
#             nn.init.xavier_uniform_(self.Wqs[i].weight)
#             nn.init.xavier_uniform_(self.Wks[i].weight)
#             nn.init.xavier_uniform_(self.Wvs[i].weight)
# 
#         self.attention_head_projection = nn.Linear(args.kqv_dim * args.n_head, 2 * args.hidden_size)
#         nn.init.kaiming_uniform_(self.attention_head_projection.weight)
#         
#         self.norm_mh = nn.LayerNorm(2 * args.hidden_size)
#         
#     def forward(self, x):
#         print(x.size())
#         WQs, WKs, WVs = [], [], []
#         
#         for i in range(args.n_head):
# 
#             WQs.append(self.Wqs[i](x))
#             WKs.append(self.Wks[i](x))
#             WVs.append(self.Wvs[i](x))
#         
#         heads = []
#         for i in range(args.n_head):
#             out = torch.bmm(WQs[i], WKs[i].transpose(1, 2))
#             out = torch.mul(out, 1 / math.sqrt(args.kqv_dim))
#             out = F.softmax(out, dim=2)
#             head_i = torch.bmm(out, WVs[i])
#             heads.append(head_i)
#         head = torch.cat(heads, dim=2)
# 
#         out = self.attention_head_projection(head)
#         out = self.norm_mh(torch.add(out, x))
#         
#         return out
# =============================================================================


class CQAttention(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = dropout
        w = torch.empty(d_model*3)
        lim = 1 / d_model
        nn.init.uniform_(w, -math.sqrt(lim), math.sqrt(lim))
        self.w = nn.Parameter(w)
        
    def forward(self, C, Q, cmask, qmask):
        # C: batch_size x seq_len_c x hid_dim
        # Q: batch_size x seq_len_q x hid_dim
        shape = (C.size(0), C.size(1), Q.size(1), C.size(2))
        Ct = C.unsqueeze(2).expand(shape)
        Qt = Q.unsqueeze(1).expand(shape)
        
        CQ = torch.mul(Ct, Qt)
        S = torch.cat([Ct, Qt, CQ], dim=3)
        S = torch.matmul(S, self.w)
        
        
        cmask = cmask.squeeze(2).permute(0, 2, 1)
        qmask = qmask.squeeze(1)
        
        S1 = S.masked_fill(cmask==0, -1e10)
        S1 = F.softmax(S1, dim = 2)
        
        S2 = S.masked_fill(qmask==0, -1e10)
        S2 = F.softmax(S2, dim = 1)
        
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1,2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        out = F.dropout(out, p=self.dropout, training=self.training)
        
        return out

class EncoderBlock(nn.Module):
    def __init__(self, conv_num, d_model, k, max_length, n_head, dropout, device):
        super().__init__()
        self.conv_num = conv_num
        self.d_model = d_model
        self.n_head = n_head
        # pos encoding
        self.pos_embedding = nn.Embedding(max_length, self.d_model)
        self.device = device
        
        # conv block 
        self.normb = nn.LayerNorm(self.d_model)
        self.convs = nn.ModuleList([DepthwiseSeparableConv(self.d_model, self.d_model, k) for _ in range(conv_num)])
        self.norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(conv_num)])
        self.dropout = dropout
        
        # self attention
        self.self_att = MultiHeadAttention(self.d_model, self.n_head, self.dropout, self.device)
        self.norme = nn.LayerNorm(self.d_model)
        
        # feedforward
        self.fc = nn.Linear(self.d_model, self.d_model, bias=True)
    
    def forward(self, x, mask):
        # x = batch_size x d_model x seq_len
        x = x.permute(0, 2, 1)
        # x = batch_size x seq_len x d_model
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        pos = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        pos_emb = self.pos_embedding(pos)
        out = x + pos_emb
        
        res = out.permute(0, 2, 1)
        out = self.normb(out)
        for i, conv in enumerate(self.convs):
            out = conv(out.permute(0, 2, 1))
            out = F.relu(out)
            out = out + res
            if (i + 1) % 2 == 0:
                p_drop = self.dropout * (i+1) / self.conv_num
                out = F.dropout(out, p=p_drop, training=self.training)
            res = out
            out = self.norms[i](out.permute(0, 2, 1))
        # out = batch_size x seq_len x d_model
        
        out, _ = self.self_att(out, out, out, mask)
        out = out + res.permute(0, 2, 1)
        # out = batch_size x seq_len x d_model
        
        out = F.dropout(out, p=self.dropout, training=self.training)
        res = out
        out = self.norme(out)
        
        out = self.fc(out)
        out = F.relu(out)
        
        out = out + res
        out = F.dropout(out, p=self.dropout, training=self.training)
        
        return out

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(d_model * 4)
        self.enc_attn_layer_norm = nn.LayerNorm(d_model * 4)
        self.device=device
        
        self.ff_layer_norm = nn.LayerNorm(d_model * 4)
        
        self.self_attention = MultiHeadAttention(hid_dim=d_model*4, n_head=n_head, dropout=dropout, device=device)
        self.encoder_attention = MultiHeadAttention(hid_dim=d_model*4, n_head=n_head, dropout=dropout, device=device)
        
        self.fc = nn.Linear(d_model*4, d_model*4, bias=True)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, question_emb, enc_emb, question_mask, enc_mask):
        """
        question: batch_size, question_len, hid_dim ----- dim !!!!!!!!!!!!
        enc_emb: batch_size, seq_len, hid_dim x 4
        question_mask: batch_size, question_len
        enc_mask: batch_size, seq_len
        """
        
        _question, _ = self.self_attention(question_emb, question_emb, question_emb, question_mask)
        
        # F.dropout(_question, p=self.dropout, training=self.training)
        question = self.self_attn_layer_norm(question_emb + self.dropout(_question))
        # batch_size x question_len x hid_dim
        
        _question, attention = self.encoder_attention(question_emb, enc_emb, enc_emb, enc_mask)
        # batch_size x question_len x hid_dim

        question = self.enc_attn_layer_norm(question + self.dropout(_question))
        
        _question = self.fc(question)
        _question = F.relu(_question)
        
        question = self.ff_layer_norm(question + self.dropout(_question))
        
        return question, attention
    
class Decoder(nn.Module):
    def __init__(self, output_dim, n_layers, hidden_size, d_model, n_head, dropout, max_length, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.max_length = max_length
        self.device = device
        
        self.pos_embedding = nn.Embedding(self.max_length, hidden_size*2)
        
        self.fc = nn.Linear(hidden_size*2, d_model*n_head, bias=True)
        
        self.layers = nn.ModuleList([DecoderLayer(d_model, 
                                                  n_head,  
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(d_model*4, output_dim)
    
    def forward(self, question_emb, enc_emb, question_mask, enc_mask):
        # question: batch_size, question_len x hidden_size*2
        batch_size = question_emb.size(0)
        question_len = question_emb.size(1)
        
        pos = torch.arange(0, question_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        pos_emb = self.pos_embedding(pos)
        ques_emb = question_emb + pos_emb
        
        ques_emb = F.relu(self.fc(ques_emb))
        ques_emb = self.dropout(ques_emb)
        
        for layer in self.layers:
            ques_emb, attention = layer(ques_emb, enc_emb, question_mask, enc_mask) 
        
        output = self.fc_out(ques_emb)
        
        return output, attention

class Baseline(nn.Module):
    def __init__(self, args, pretrained):
        super(Baseline, self).__init__()
        self.args = args
        self.device = args.device
        # embedding layer
        self.emb = Embedding(args, pretrained).to(self.device)
        
        # Depth wise separable conv layer
        self.context_conv = DepthwiseSeparableConv(self.args.word_dim + self.args.char_channel_size, args.d_model, args.kernel_size).to(self.device)
        self.answer_conv = DepthwiseSeparableConv(self.args.word_dim + self.args.char_channel_size, args.d_model, args.kernel_size).to(self.device)
        
        # multihead self attention
        self.c_enc = EncoderBlock(conv_num=args.conv_num, d_model=args.d_model, k=args.kernel_size, max_length=args.max_len_context+2, n_head=args.n_head, dropout=args.dropout, device=args.device).to(self.device)
        self.a_enc = EncoderBlock(conv_num=args.conv_num, d_model=args.d_model, k=args.kernel_size, max_length=args.max_len_answer+5, n_head=args.n_head, dropout=args.dropout, device=args.device).to(self.device)
        
        # query and context attention
        self.ca_att = CQAttention(d_model=args.d_model, dropout=args.dropout).to(self.device)
        
        # decoder
        self.decoder = Decoder(output_dim=args.output_dim, n_layers=args.DEC_LAYERS, hidden_size=args.hidden_size, d_model=args.d_model, n_head=args.n_head, dropout=args.dropout, max_length=args.max_len_question+2, device=self.device).to(self.device)
    
    def make_enc_mask(self, src):
        c_mask = (src != self.args.pad_idx_encoder).unsqueeze(1).unsqueeze(2)
        
        # batch_size  x 1 x 1 x seq_len
        # mask on the last dimension
        return c_mask
    
    def make_dec_mask(self, trg):
        trg_pad_mask = (trg != self.args.pad_idx_encoder).unsqueeze(1).unsqueeze(2)
        # batch_size  x 1 x 1 x seq_len
        
        seq_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((seq_len, seq_len))).bool().to(self.device)
        # seq_len x seq_len
        
        trg_mask = trg_pad_mask & trg_sub_mask
        # batch_size, 1, seq_len, seq_len
        
        return trg_mask
    
    def forward(self, context_word, context_char, answer_word, answer_char, question_word, question_char):
        # self, batch
        cmask = self.make_enc_mask(context_word).to(self.device)
        amask = self.make_enc_mask(answer_word).to(self.device)
        # emb size: bath x seq_len x (word_dim + char_channel_size)
        C_emb = self.emb(context_char, context_word)
        A_emb = self.emb(answer_char, answer_word)
        
        # encoding after conv batch x d_model x seq_len
        C = self.context_conv(C_emb.permute(0, 2, 1))
        A = self.answer_conv(A_emb.permute(0, 2, 1))
        
        Ce = self.c_enc(C, cmask)
        Ae = self.a_enc(A, amask)
        
        encoded = self.ca_att(Ce, Ae, cmask, amask)
        
        trg_mask = self.make_dec_mask(question_word).to(self.device)
        
        Q_emb = self.emb(question_char, question_word)
        output, attention = self.decoder(Q_emb, encoded, trg_mask, cmask)
        
        return output, attention
#self.pos_emb = nn.Embedding()