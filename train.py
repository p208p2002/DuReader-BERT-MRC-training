from transformers import AlbertForMaskedLM,AlbertConfig
from lib.tokenizer import init_tokenizer
from lib.Dataset import DURDataset
from torch.utils.data import DataLoader
import torch 
from transformers import AdamW


if __name__ == "__main__":
    tokenizer = init_tokenizer("voidful/albert_chinese_tiny")
    # config = AlbertConfig.from_pretrained('voidful/albert_chinese_tiny')
    # config.update({'vocab_size':config.vocab_size+2})
    # print(config)
    # exit()
    model = AlbertForMaskedLM.from_pretrained("voidful/albert_chinese_tiny")
    train_dataset = DURDataset(tokenizer,'training_dataset/dev.data.cht.txt')
    train_dataloader = DataLoader(train_dataset,batch_size=16,shuffle=True)

    # setting device    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("using device",device)
    model.to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6, eps=1e-8)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    model.zero_grad()
    for epoch in range(30):
        running_loss_val = 0.0
        running_acc = 0.0
        for batch_index, batch_dict in enumerate(train_dataloader):
            model.train()
            
            batch_dict = [t.to(device) for t in batch_dict]

            # print("======000000======",batch_dict[0])
            # print("======111111======",batch_dict[1])
            # print("======222222======",batch_dict[2])
            # print("======333333======",batch_dict[3])
            
            outputs = model(
                batch_dict[0],
                # token_type_ids = batch_dict[1],
                # attention_mask = batch_dict[2],
                # labels = batch_dict[3]
                )
            # loss,logits = outputs[:2]
            # loss.sum().backward()
            # optimizer.step()
            # # scheduler.step()  # Update learning rate schedule
            # model.zero_grad()

            # # compute the loss
            # loss_t = loss.item()
            # running_loss_val += (loss_t - running_loss_val) / (batch_index + 1)

            # # compute the accuracy
            # acc_t = compute_accuracy(logits, batch_dict[3])
            # running_acc += (acc_t - running_acc) / (batch_index + 1)

            # # log
            # print("epoch:%2d batch:%4d train_loss:%2.4f train_acc:%3.4f"%(epoch+1, batch_index+1, running_loss_val, running_acc))