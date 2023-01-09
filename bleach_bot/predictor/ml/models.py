import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import Linear


def binary_accuracy(y_hat, y_test):
    y_hat = torch.round(torch.sigmoid(y_hat))
    correct_sum = (y_hat == y_test).sum().float()
    acc = correct_sum / y_test.shape[0]
    acc = acc * 100
    return acc


class TextBinaryClassifier(pl.LightningModule):
    def __init__(
        self,
        vocab_size,
        channels=256,
        dropout=0.4,
        lr=1e-4,
    ):
        super().__init__()

        self.lr = lr
        self.dropout = dropout
        self.vocab_size = vocab_size

        self.embeddings = torch.nn.Embedding(self.vocab_size, embedding_dim=channels)

        self.pos_embedding = torch.nn.Embedding(1024, embedding_dim=channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, nhead=4, dropout=self.dropout, dim_feedforward=1024
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=8)

        self.linear = Linear(channels, 1)

        self.do = nn.Dropout(p=self.dropout)

        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        batch_size, sequence_len = x.size(0), x.size(1)

        embedded = self.embeddings(x)

        pos_x = (
            torch.arange(0, sequence_len, device=x.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        pos_x = self.pos_embedding(pos_x)

        embedded += pos_x

        embedded = self.do(embedded)

        embedded = embedded.permute(1, 0, 2)

        transformed = self.encoder(embedded)

        transformed = transformed.permute(1, 0, 2)

        out = self.linear(transformed[:, 0])

        return out

    def _step(self, batch, batch_idx, name):
        x, y = batch

        y_hat = self(x)

        y_hat = y_hat.view(-1)
        y = y.view(-1)

        loss = self.loss(y_hat, y)
        accuracy = binary_accuracy(y_hat, y)

        self.log(f"{name}_loss", loss)
        self.log(f"{name}_accuracy", accuracy)

        return loss

    def training_step(self, batch, batch_idx):

        return self._step(batch, batch_idx, name="train")

    def validation_step(self, batch, batch_idx):

        return self._step(batch, batch_idx, name="valid")

    def test_step(self, batch, batch_idx):

        return self._step(batch, batch_idx, name="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }


if __name__ == "__main__":
    max_vocab_size = 30_000

    tokens = torch.randint(low=0, high=max_vocab_size, size=(20, 32))

    text_classifier = TextBinaryClassifier(vocab_size=max_vocab_size)

    output = text_classifier(tokens)

    print(output.shape)
    print(output)
