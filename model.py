import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import pickle

device = torch.device("cpu")

#辞書の読み込み
with open('vocab_en.pkl', "rb") as f:
    vocab_en = pickle.load(f)

with open('vocab_ja.pkl', "rb") as f:
    vocab_ja = pickle.load(f)

import spacy

JA = spacy.load('ja_core_news_md')
EN = spacy.load('en_core_web_md')

def tokenize_ja(sentence):
    return [tok.text for tok in JA.tokenizer(sentence)]

def tokenize_en(sentence):
    return [tok.text for tok in EN.tokenizer(sentence)]

import math

class PositionalEncoder(pl.LightningModule):

    def __init__(self, d_model=512, max_seq_len=110):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.d_model = d_model

        # 0 の行列を作成（Sequence_length, Embedding_dim）
        pe = torch.zeros(max_seq_len, d_model)

        # pe に位置情報が入った配列を追加
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):

                # 配列中の0 と偶数インデックスには sin 波を適用
                pe[pos, i] = math.sin(pos / 10000.0 ** ((2 * i) / d_model))

                # 配列中の奇数インデックスには cos 波を適用
                pe[pos, i + 1] = math.cos(pos / 10000.0 ** ((2 * (i + 1)) / d_model))

        # PE を pe という名前でモデルに保存
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 埋め込み表現の値に sqrt を掛け値を大きくする
        x = x * math.sqrt(self.d_model)

        # 元の埋め込み表現に pe を足し合わせ位置情報を付加
        x = x + self.pe[:x.size(1), :]
        x = self.dropout(x)
        return x

class Encoder(pl.LightningModule):

    def __init__(self, src_vocab_length, d_model, nhead, dim_feedforward, num_encoder_layers, dropout, activation):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_length, d_model)
        self.pos_encoder = PositionalEncoder(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)


    def forward(self, src, src_pad_mask):

        src_embeded = self.src_embedding(src)
        pos_src = self.pos_encoder(src_embeded)
        memory = self.encoder(pos_src, src_key_padding_mask=src_pad_mask)

        return memory

class Decoder(pl.LightningModule):

    def __init__(self, trg_vocab_length, d_model, nhead, dim_feedforward, num_decoder_layers, dropout, activation):
        super().__init__()

        self.trg_embedding = nn.Embedding(trg_vocab_length, d_model)
        self.pos_encoder = PositionalEncoder(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

    def forward(self, memory, trg_input, trg_mask, trg_pad_mask):

        trg_embeded = self.trg_embedding(trg_input)
        pos_trg = self.pos_encoder(trg_embeded)
        output = self.decoder(pos_trg, memory, tgt_mask=trg_mask, tgt_key_padding_mask=trg_pad_mask)

        return output

class Transformer(pl.LightningModule):

    def __init__(self, src_vocab_length, trg_vocab_length, src_pad_idx, trg_pad_idx, *arg):
        super().__init__()

        self.src_vocab_length=src_vocab_length
        self.trg_vocab_length=trg_vocab_length
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(
            self.src_vocab_length, d_model=512, nhead=8, dim_feedforward=2048,
            num_encoder_layers=6, dropout=0.1, activation='relu'
        )
        self.decoder = Decoder(
            self.trg_vocab_length, d_model=512, nhead=8, dim_feedforward=2048,
            num_decoder_layers=6, dropout=0.6, activation='relu'
        )

        self.out = nn.Linear(512, self.trg_vocab_length)

    def create_pad_mask(self, input_word, pad_idx):
        pad_mask = input_word == pad_idx
        return pad_mask

    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0)).to(device)
        return mask

    def forward(self, src, trg):
        trg_input = trg[:, :-1]

        # 各種 Mask
        src_pad_mask = self.create_pad_mask(src, self.src_pad_idx)
        trg_pad_mask = self.create_pad_mask(trg_input, self.trg_pad_idx)
        trg_mask = self.generate_square_subsequent_mask(trg_input.size(1))

        memory = self.encoder(src, src_pad_mask)
        output = self.decoder(memory, trg_input, trg_mask, trg_pad_mask)

        logit = self.out(output)
        return logit

    def training_step(self, batch, batch_idx):
        src, trg = batch

        logit = self(src, trg)

        targets = trg[:, 1:].reshape(-1)
        logit = logit.view(-1, logit.size(-1))

        loss = F.cross_entropy(logit, targets, ignore_index=self.trg_pad_idx)
        self.log('train_loss', loss, logger=True, on_step=False, on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx):
        src, trg = batch

        logit = self(src, trg)

        targets = trg[:, 1:].reshape(-1)
        logit = logit.view(-1, logit.size(-1))

        loss = F.cross_entropy(logit, targets, ignore_index=self.trg_pad_idx)
        self.log('val_loss', loss,  logger=True, on_step=False, on_epoch=True)

        return loss


    def test_step(self, batch, batch_idx):
        src, trg = batch

        logit = self(src, trg)

        targets = trg[:, 1:].reshape(-1)
        logit = logit.view(-1, logit.size(-1))

        loss = F.cross_entropy(logit, targets, ignore_index=self.trg_pad_idx)
        self.log('test_loss', loss,  logger=True, on_step=False, on_epoch=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        return optimizer

