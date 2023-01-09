import math
import numpy as np
import onnxruntime
from tokenizers import Tokenizer


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class TextAudit:
    def __init__(self):
        self.inference_session = onnxruntime.InferenceSession("data/toxicity_model.onnx")
        self.tokenizer = Tokenizer.from_file("data/tokenizer.json")

    def predict(self, text):
        input_name = self.inference_session.get_inputs()[0].name
        label_name = self.inference_session.get_outputs()[0].name

        ids_l = self.tokenizer.encode(text).ids

        prediction = self.inference_session.run(
            [label_name], {input_name: np.array([ids_l]).astype(np.int32)}
        )[0]

        return sigmoid(prediction.item()) < 0.8


audit = TextAudit()
if __name__ == "__main__":
    while True:
        print(audit.predict(input("> ")))
