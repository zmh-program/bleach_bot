import time

import numpy as np
import onnxruntime as rt
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from bleach_bot.predictor.ml.models import TextBinaryClassifier


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="../../data/toxicity_model.ckpt")
    parser.add_argument("--output_path", default="../../data/toxicity_model.onnx")
    parser.add_argument("--tokenizer", default="../../data/tokenizer.json")
    args = parser.parse_args()

    model_path = args.model_path
    tokenizer = Tokenizer.from_file(args.tokenizer)
    filepath = args.output_path

    text = "f***k youu"

    model = TextBinaryClassifier(
        vocab_size=tokenizer.get_vocab_size(),
        lr=1e-4,
        dropout=0.1,
    )

    model.load_state_dict(torch.load(model_path)["state_dict"])

    model.eval()

    ids_l = tokenizer.encode(text).ids

    ids = torch.tensor(ids_l, dtype=torch.int).unsqueeze(0)

    start = time.time()

    for _ in tqdm(range(100)):
        with torch.no_grad():
            out = model(ids)

    print(sigmoid(out.numpy()))

    print(f"Duration Torch for 100 predictions: {time.time() - start}")

    input_sample = ids

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

    sess = rt.InferenceSession(filepath)

    start = time.time()

    for _ in tqdm(range(100)):
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        pred = sess.run([label_name], {input_name: np.array([ids_l]).astype(np.int32)})[
            0
        ]

    print(sigmoid(pred), type(pred))

    print(f"Duration ONNX for 100 predictions: {time.time() - start}")
