from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace, Sequence, Digits, Punctuation
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
            "/nlp/reddit_full_text.txt",
            "/nlp/amazon_full_text.txt",
        ],
    )
    parser.add_argument("--output_path", default="../../data/tokenizer.json")
    args = parser.parse_args()

    files = args.files
    output_path = args.output_path

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    tokenizer.pre_tokenizer = Sequence(
        [Whitespace(), Digits(individual_digits=False), Punctuation()]
    )

    tokenizer.normalizer = Lowercase()

    tokenizer.train(files, trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )

    output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
    print(output)

    print(output.tokens)
    print(output.ids)
    print(output.offsets)
    print(output.overflowing)

    tokenizer.save(output_path, pretty=True)

    tokenizer = Tokenizer.from_file(output_path)

    output = tokenizer.encode("Hello, y'all! How are you 😁 ?")

    print(output.tokens)
    print(output.ids)
    print(output.offsets)
    print(output.overflowing)
