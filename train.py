from transformers import AlbertForMaskedLM
from lib.tokenizer import init_tokenizer
from lib.Dataset import DURDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    tokenizer = init_tokenizer("voidful/albert_chinese_tiny")
    model = AlbertForMaskedLM.from_pretrained("voidful/albert_chinese_tiny")
    train_dataset = DURDataset(tokenizer,'training_dataset/dev.data.cht.txt')
    training_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True)