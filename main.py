from fastapi import FastAPI, HTTPException
import torch
from model import Transformer  # モデルクラスのインポート

#model.py で定義されているトークナイザーと辞書をimport
from model import tokenize_ja, tokenize_en, vocab_en, vocab_ja

# モデルとトークナイザの読み込み
src_vocab_length = len(vocab_en)
trg_vocab_length = len(vocab_ja)
src_pad_idx = vocab_en['<pad>']
trg_pad_idx = vocab_ja['<pad>']

model = Transformer.load_from_checkpoint(
    checkpoint_path='epoch=4-step=40950.ckpt',
    src_vocab_length=src_vocab_length,
    trg_vocab_length=trg_vocab_length,
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx
)
model.eval()
device = torch.device("cpu")

# 英語から日本語への翻訳を行う関数
def perform_translation(text, model=model, vocab_en=vocab_en, vocab_ja=vocab_ja, device=device, max_len=110):
    # 分かち書きしていない英語のテキストをトークン化し、インデックスに変換
    src = tokenize_en(text)
    src = torch.LongTensor([[vocab_en[tok] for tok in src]]).to(device)

    src_pad_mask = src == vocab_en['<pad>']

    # Encoder
    memory = model.encoder(src, src_pad_mask)

    # Decoder への入力の取得
    trg = torch.LongTensor([vocab_ja['<sos>']]).unsqueeze(1).to(device)

    translated_sentence = ''

    for i in range(1, max_len):
        # 推論時にその時点での Decoder への入力文字列分の Mask を毎度作成
        size = trg.size(1)
        trg_mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        trg_mask = trg_mask.float().masked_fill(trg_mask==0, float('-inf')).masked_fill(trg_mask==1, float(0.0)).to(device)

        trg_pad_mask = trg == vocab_ja['<pad>']

        # 推論
        dec_out = model.decoder(memory, trg, trg_mask, trg_pad_mask)
        pred = model.out(dec_out)

        # 推論の結果を文字列として追加
        add_word = vocab_ja.lookup_token(pred.argmax(dim=-1)[-1][i-1])
        translated_sentence += " "+ add_word

        # <eos> が予測されたら推論を修了
        if add_word=='<eos>':
            break

        # 推論結果を trg に結合（cat）
        trg = torch.cat((trg, torch.LongTensor([[pred.argmax(dim=-1)[-1][i-1]]]).to(device)), dim=1)

    return translated_sentence.replace("<sos>", "").replace("<eos>", "").strip()

