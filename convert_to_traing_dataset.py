import os
from lib.tokenizer import init_tokenizer
from lib.data_procress import load_data_from_mrc as load_data
from lib.special_token_def import SEP_ANSWER_START,SEP_ANSWER_END,SEP,CLS

def make_training_data_set(f,mrc_csv_paths):
    #
    max_length = 384 # Q + C 長度限制
    stride = int(max_length/2)
    number_of_special_tokens = len([CLS,SEP,SEP,SEP_ANSWER_START,SEP_ANSWER_END])
    max_ans_length = 512 - max_length - number_of_special_tokens
    
    tokenizer = init_tokenizer("voidful/albert_chinese_tiny")

    for mrc_csv_path in mrc_csv_paths:
        for data in load_data(mrc_csv_path):
            #
            try:
                q,a,c = data
            except Exception as e:
                print(e)
                continue
            
            q_tokens = tokenizer.tokenize(q)
            c_tokens = tokenizer.tokenize(c)
            a_tokens = tokenizer.tokenize(a)[:max_ans_length]

            context_padding = 0
            _max_context_length = max_length - len(q_tokens)
            while(True):
                #
                _c_tokens = c_tokens[0+context_padding:_max_context_length+context_padding]
                current_total_length = len(a_tokens)+len(_c_tokens)+len(q_tokens)+number_of_special_tokens
                assert current_total_length <= 512
                if(len(_c_tokens)<=0 or len(q_tokens)+len(_c_tokens) <= stride):
                    break
                
                # print(len(_c_tokens),current_total_length)
                question_string = tokenizer.convert_tokens_to_string(q_tokens).replace(" ","")
                context_string = tokenizer.convert_tokens_to_string(_c_tokens).replace(" ","")
                answer_string = tokenizer.convert_tokens_to_string(a_tokens).replace(" ","")
                f.write(CLS+question_string+SEP+context_string+SEP+SEP_ANSWER_START+answer_string+SEP_ANSWER_END+"\n")

                #
                context_padding += stride

if __name__ == "__main__":
    os.system('rm -rf training_dataset/&&mkdir training_dataset')

    with open("training_dataset/dev.data.cht.txt",'w',encoding = 'utf-8') as f:
        # make_training_data_set(f,['mrc_dataset/mrc_search.dev.cht.csv','mrc_dataset/mrc_zhidao.dev.cht.csv'])
        make_training_data_set(f,['mrc_dataset/mrc_search.dev.cht.csv'])

    with open("training_dataset/train.data.cht.txt",'w',encoding = 'utf-8') as f:
        make_training_data_set(f,['mrc_dataset/mrc_search.train.cht.csv','mrc_dataset/mrc_zhidao.train.cht.csv'])
    
    