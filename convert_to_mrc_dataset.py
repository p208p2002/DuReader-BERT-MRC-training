from lib.data_procress import load_data_from_raw as load_data
from lib.MLDataModel import MLDataModel
from lib.special_token_def import NO_ANSWER
from tqdm import tqdm
import os

def write_file(f,q,a,c,separate=','):
    f.write(q+separate+a+separate+c+'\n')

def convert_to_mrc_foramt(dataset_path):    
    output_file_name = 'mrc_dataset/mrc_%s.csv'%os.path.basename(dataset_path).replace('.json','')
    output_file = open(output_file_name,'w',encoding='utf-8')

    daset_total_count = len(open(dataset_path).readlines())
    pbar = tqdm(total=daset_total_count)
    for data in load_data(dataset_path):
        question = data['question']
        answers_current_index = 0
        answers = data['answers']
        
        for document in data['documents']:
            paragraph = ''.join(document['paragraphs'])
            paragraph = paragraph.replace("\n","")
            paragraph = paragraph.replace(",","，")
            has_ans = document['is_selected']

            if(has_ans):
                try:
                    answer = answers[answers_current_index]
                    # print("C:%s, Q:%s, A:%s"%(paragraph[:20],question,answers[answers_current_index]))
                    write_file(output_file,q=question,a=answer,c=paragraph)
                    answers_current_index += 1
                except:
                    pass
            else:
                # print("C:%s, Q:%s, A:%s"%(paragraph[:20],question,NO_ANSWER))
                write_file(output_file,q=question,a=NO_ANSWER,c=paragraph)
                pass
            
            
        pbar.update(1)
    pbar.close()
    output_file.close()

if __name__ == "__main__":
    os.system('rm -rf mrc_dataset/&&mkdir mrc_dataset')
    dataset_paths = [
        "./dataset/devset/search.dev.cht.json",
        "./dataset/devset/zhidao.dev.json",
        # "./dataset/testset/search.test.json",
        # "./dataset/testset/zhidao.test.json",
        "./dataset/trainset/search.train.json",
        "./dataset/trainset/zhidao.train.json" 
    ]
    for dataset_path in dataset_paths:
        convert_to_mrc_foramt(dataset_path)