from transformers import BertTokenizerFast
from .special_token_def import NO_ANSWER
def init_tokenizer(model_name_or_vocab_path):
    tokenizer = BertTokenizer.from_pretrained(model_name_or_vocab_path)
    tokenizer._add_tokens([NO_ANSWER],special_tokens=True)
    return tokenizer
