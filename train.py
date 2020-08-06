from transformers import AlbertForMaskedLM
from lib.tokenizer import init_tokenizer

if __name__ == "__main__":
    tokenizer = init_tokenizer("voidful/albert_chinese_tiny")
    # model = AlbertForMaskedLM.from_pretrained("voidful/albert_chinese_tiny")