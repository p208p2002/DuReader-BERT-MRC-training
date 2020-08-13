from lib.tokenizer import init_tokenizer
from lib.data_procress import load_data_from_mrc as load_data
if __name__ == "__main__":
    
    qc_length_limit = 400
    tokenizer = init_tokenizer("voidful/albert_chinese_tiny")
    for data in load_data('mrc_dataset/mrc_search.dev.cht.csv'):
        pass
        # print(data)
        # tokenizer.
