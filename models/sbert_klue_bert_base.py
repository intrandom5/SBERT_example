from transformers import AutoModel
import torch.nn as nn
import torch


class SBERT_with_KLUE_BERT(nn.Module):
    def __init__(self):
        super(SBERT_with_KLUE_BERT, self).__init__()
        self.bert = AutoModel.from_pretrained("klue/bert-base")
        self.similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, src_ids, tgt_ids):
        u = self.bert(src_ids)['pooler_output']
        v = self.bert(tgt_ids)['pooler_output']

        outputs = self.similarity(u, v)

        return outputs
