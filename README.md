# bleach_bot

A bot that can detect negative or toxic messages

## Data:

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

## Train the tokenizer

```
python bleach_bot/ml/train_tokenizer.py --files YOUR_TEXT_FILES
```

## Train the model

```
python bleach_bot/ml/train_toxicity_classifier.py --tokenizer data/tokenizer.json \
                                                  --data_path data/train.csv
```

## Convert the model to onnx

```
python bleach_bot/ml/torch_to_onnx.py --tokenizer data/tokenizer.json \
                                      --model_path data/toxicity_model.ckpt \
                                      --output_path data/toxicity_model.onnx
```

# Description

### Introduction

Content moderation can be difficult given the scale at which text is generated
by users of the internet. One solution to simplify this process is to automate
it using machine learning. An ML model trained on examples of what the moderator
does not want to see, like toxic content, insults, or racist comments can then
be used to automatically filter those messages out.

In this project, we are going to train such model using Jigsaw Toxic Comment
Data-set:
[https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)

### Data

The Jigsaw toxicity data includes 159,000 samples, each sample can be labeled
with multiple categories like “toxic”, “insult”…

![](https://cdn-images-1.medium.com/max/800/1*0ek-3vfHzNtfb1S5BA-bBw.png)
<span class="figcaption_hack">Dataset format — image by author</span>

For simplicity, we use all those categories to create a single binary target as
follows :

data["label"] = (
    data[
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ].sum(axis=1, skipna=True)
    > 0.5
).astype(int)
### Machine Learning steps

![](https://cdn-images-1.medium.com/max/800/1*cDb11CVFNraftKnD6yQ3Mg.png)
<span class="figcaption_hack">Machine learning steps — image by author</span>

#### Tokenizer

I trained my own BPE tokenizer using huggingface’s library, you can do the same
using the script in my Github repository:

python bleach_bot/ml/train_tokenizer.py --files YOUR_TEXT_FILES
This tokenizer breaks the sentence into small tokens and then maps each of them
to integers:

![](https://cdn-images-1.medium.com/max/1200/1*c4ROlohhaWMhp62tvXslBQ.png)
<span class="figcaption_hack">Tokenizer — image by author</span>

#### Classifier

We use a transformer network as a classifier:

![](https://cdn-images-1.medium.com/max/800/1*7ExydVQu24lNxkGr5On_9g.png)
<span class="figcaption_hack">Transformer network — Image by author</span>

The implementation is made easy by using the torch.nn.TransformerEncoderLayer
and torch.nn.TransformerEncoder classes.

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
### Predictor

#### **Torch to onnx**

For practical reasons, we convert the model from torch .ckpt format to
.onnx.<br> We also use the onnxruntime library to use this model in our
prediction.

To do that we run:

torch.onnx.export(
    model,  # model being run
    ids,  # model input (or a tuple for multiple inputs)
    filepath,  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={
        "input": {0: "batch_size", 1: "sequence_len"},  # variable length axes
        "output": {0: "batch_size"},
    },
)
Doing this process allows to reduce the size of the model by 66% and improve the
prediction speed on CPU by 68% ( from 2.63ms to 0.85ms to produce a prediction
for a small sentence).

#### Prediction Servers

We use a queuing system with RabbitMQ and pika to process prediction queries
coming from the bot.

![](https://cdn-images-1.medium.com/max/800/1*A9j6OC7XiFY9pazp2i3yVw.png)
<span class="figcaption_hack">Prediction architecture — Image by author</span>

This architecture allows to isolate the bot logic from the machine learning/NLP
logic and would make it easier to horizontal scale to multiple predictors if
needed.

You can run this whole architecture using the docker-compose file from my
repository:

First, get your bot token by following this tutorial:

Then, download the model and tokenizer:

wget https://github.com/CVxTz/bleach_bot/releases/download/v1/toxicity_model.onnx -P ./data/

wget https://github.com/CVxTz/bleach_bot/releases/download/v1/tokenizer.json -P ./data/
Finally, run docker-compose

docker-compose up --build
### Bot Demo

The bot deletes all the messages that are given a score greater than 0.8 by the
classification model.

Next is the demo. I run the bot on my machine using docker-compose. We can see
that the bot deletes all the nasty and negative messages and keeps the regular
ones. Don’t blink, because it is really fast 😉

![](https://cdn-images-1.medium.com/max/800/1*niVCXXY9C7OUG5KvDATKYA.gif)

### Conclusion

This project details the first steps needed to build a moderation bot using deep
learning. The bot is trained to detect toxic or insulting messages and to
automatically delete them. The next steps would be to further improve the
Machine learning part of the bot to reduce the number of false positives and
also to work on its deployment.
