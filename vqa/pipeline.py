from efficientnet_pytorch import EfficientNet
from torch import nn
from transformers import AutoModel, AutoTokenizer


class VQAPipeline(nn.Module):
    def __init__(
        self,
        cnn_model="efficientnet-b0",
        bert_model="bert-base-uncased",
        hidden_size=1000,
        num_labels=430,
        dropout=0.5,
    ):
        super().__init__()
        self.cnn = EfficientNet.from_pretrained(cnn_model, include_top=False)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert = AutoModel.from_pretrained(bert_model)
        self.decoder = nn.Sequential(
            nn.Linear(1280 + 768, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, images, questions):
        cnn_out = self.cnn(images).squeeze(-1).squeeze(-1)
        # cnn_out.shape == (batch_size, 1280)
        tokens = self.tokenizer(
            questions, padding=True, truncation=True, return_tensors="pt"
        )
        bert_out = self.bert(**tokens)
        cls_tokens = bert_out["last_hidden_state"][:, 0, :]
        # cls_tokens.shape == (batch_size, 768)
        combined = torch.cat((cnn_out, cls_tokens), dim=-1)
        # combined.shape == (batch_size, 1280 + 768)
        out = self.decoder(combined)
        # out.shape == (batch_size, 1000)
        return out


def toggle_freeze(model, do_freeze=True):
    for p in model.parameters():
        p.requires_grad = do_freeze


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
