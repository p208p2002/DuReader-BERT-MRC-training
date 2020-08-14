from torch.utils.data.dataset import Dataset
from .special_token_def import SEP, SEP_ANSWER_START, SEP_ANSWER_END, MASK, PAD
from random import randint
import torch

class DURDataset(Dataset):
    def __init__(self,tokenizer,dataset_file_path):
        self.tokenizer = tokenizer
        self.dataset_file_path = dataset_file_path
        self.data_lines = open(dataset_file_path).readlines()
        
    def __getitem__(self, index):
        data_line = self.data_lines[index][:].replace('\n','')
        data_line = data_line.split(SEP)
        assert len(data_line) == 3
        q,c,a = data_line
        answer_span = a.replace(SEP_ANSWER_START,"")
        answer_span_tokens = self.tokenizer.tokenize(answer_span)
        random_mask_index = randint(0,len(answer_span_tokens)-1)
        
        #
        true_label = answer_span_tokens[random_mask_index]
        answer_span_tokens[random_mask_index] = MASK

        #
        question_tokens = self.tokenizer.tokenize(q)
        context_tokens = self.tokenizer.tokenize(c)
        answer_span_tokens = [SEP_ANSWER_START] + answer_span_tokens
        output_tokens = question_tokens + [SEP] + context_tokens + [SEP] +answer_span_tokens
        # print(output_tokens)

        #
        output_tokens_mask_index = output_tokens.index(MASK)
        output_tokens = output_tokens[:output_tokens_mask_index]
        true_label_id = self.tokenizer.convert_tokens_to_ids(true_label)

        # padding
        len_of_output_tokens_without_padding = len(output_tokens)
        while len(output_tokens)<512:
            output_tokens.append(PAD)
        assert len(output_tokens) == 512
        
        # create answer label
        tensor_answer_label = torch.LongTensor([-100]*512)
        tensor_answer_label[output_tokens_mask_index] = true_label_id

        # inputs
        input_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(output_tokens))

        token_type_ids = [0]*len(question_tokens + [SEP]) + [1]*len(context_tokens + [SEP]) + [0]*len(answer_span_tokens)
        while len(token_type_ids) < 512:
            token_type_ids.append(1)
        assert len(token_type_ids) == 512
        token_type_ids = torch.LongTensor(token_type_ids)
        
        attention_mask = [1]*len(output_tokens)
        while len(attention_mask) < 512:
            attention_mask.append(0)
        assert len(attention_mask) == 512
        attention_mask = torch.LongTensor(attention_mask)

        # return {
        #     'input_ids':input_ids,
        #     'token_type_ids':token_type_ids,
        #     'attention_mask':attention_mask,
        #     'label':tensor_answer_label
        # }

        return [input_ids,token_type_ids,attention_mask,tensor_answer_label]

    def __len__(self):
        return len(self.data_lines)

if __name__ == "__main__":
    import sys
    sys.path.append('../')
    from lib.tokenizer import init_tokenizer

    tokenizer = init_tokenizer('voidful/albert_chinese_tiny')
    dur_dataset = DURDataset(tokenizer,'../training_dataset/dev.data.cht.txt')

    print(dur_dataset.__getitem__(499))
