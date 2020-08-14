from transformers import BertTokenizer
from .special_token_def import NO_ANSWER, SEP_ANSWER_START, SEP_ANSWER_END
def init_tokenizer(model_name_or_vocab_path):
    tokenizer = BertTokenizer.from_pretrained(model_name_or_vocab_path)
    # tokenizer._add_tokens([NO_ANSWER, SEP_ANSWER_START, SEP_ANSWER_END],special_tokens=True)
    return tokenizer
