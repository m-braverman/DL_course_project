from tqdm import tqdm
from loguru import logger
import torch
from mixup_classifier import *
from sst_dataset import *
import torch.optim as optim

#https://github.com/munikarmanish/bert-sentiment/blob/master/bert_sentiment/train.py
# Medium: painless finetuning
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

def train_one_epoch(model, bert, criterion, optimizer, dataset, batch_size=32):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    model.train()
    train_loss, train_acc, count = 0.0, 0.0, 0
    for seq, attn_masks, labels in tqdm(dataloader):
        seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)
        optimizer.zero_grad()
        bert_output, _, __ = bert(seq, attn_masks)    
        cls_rep = bert_output[:, 0]
        
        logits = model(cls_rep)
        loss = criterion(logits.squeeze(-1), labels.float())

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        count += 1 
    train_loss /= count
    return model, train_loss

def mixup_train_one_epoch(model, bert, criterion, optimizer, dataset, alpha, batch_size=32):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers = 5
    )
    model.train()
    train_loss, count = 0.0, 0
    for seq, attn_masks, labels in tqdm(dataloader):
        seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)
        optimizer.zero_grad()
        bert_output, _, __= bert(seq, attn_masks)    
        cls_rep = bert_output[:, 0]
        #MixUp
        mixed_x, y_a, y_b, lam = mixup_data(cls_rep, labels, alpha)
            
        #Classifier
        pred = model(mixed_x)

        #Computing loss
        loss = mixup_criterion(criterion, pred, y_a.float(), y_b.float(), lam)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        count += 1 
    train_loss /= count
    return model, train_loss


def evaluate_one_epoch(model, bert, criterion, optimizer, dataset, batch_size=32):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers = 5
    )
    model.eval()
    loss, acc, count = 0.0, 0.0, 0
    with torch.no_grad():
        for seq, attn_masks, labels in tqdm(dataloader):
            seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)
            bert_output, _, __ = bert(seq, attn_masks)
            cls_rep = bert_output[:, 0]
            logits = model(cls_rep)
            loss += criterion(logits.squeeze(-1), labels.float()).item()
            acc += get_accuracy_from_logits(logits, labels)
            count += 1
        loss /= count
        acc /= count
    return loss, acc


def train(
    mixup=True,
    bert="bert-base-uncased",
    epochs=30,
    batch_size=32,
    save=True,
    alpha=0.1,
    noisy=False,
    finetune=False,
    additional_pretrain='',
    warm_classifier=''):
    if noisy:
            trainset = SSTDataset("noisy_train20.tsv")
    else:
            trainset = SSTDataset("train.tsv")
    testset = SSTDataset("test.tsv")
    print(device)
    
    bert = create_bert(additional_pretrain, finetune)
    bert = bert.to(device)
    
    sentiment_mix_up = MixUpSentimentClassifierLayer()
    if warm_classifier:
        sentiment_mix_up = torch.load(warm_classifier)
    sentiment_mix_up = sentiment_mix_up.to(device)

    lossfn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(sentiment_mix_up.parameters(), lr = 2e-5)
    best_train_loss = 1.
    train_losses, val_losses, test_losses = [], [], []
    test_accs = []
    
    for epoch in range(1, epochs+1):
        if mixup:
            sentiment_mix_up, train_loss = mixup_train_one_epoch(
                sentiment_mix_up, bert, lossfn, optimizer, trainset, alpha=alpha, batch_size=batch_size
            )
        else: 
            sentiment_mix_up, train_loss = train_one_epoch(
                sentiment_mix_up, bert, lossfn, optimizer, trainset, batch_size=batch_size
            )
        test_loss, test_acc = evaluate_one_epoch(
            sentiment_mix_up, bert, lossfn, optimizer, testset, batch_size=batch_size
        )
        logger.info(f"epoch={epoch}")
        logger.info(
            f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f}"
        )
        logger.info(
            f"test_acc={test_acc:.3f}"
        )
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        test_accs.append(test_acc)
        
        if save and train_loss < best_train_loss:
            best_train_loss = train_loss
            base_or_mixup = 'mixup' if mixup else 'base'
            noisy_or_clean = 'noisy' if noisy else 'clean'
            pretrain = 'pretrained' if additional_pretrain else ''
            alpha_str = str(alpha).replace('.', '') if mixup else '0'
            torch.save(sentiment_mix_up, f"imdb__{pretrain}_{base_or_mixup}__alpha{alpha_str}__{noisy_or_clean}.model")
            
    losses = {'train': train_losses,
              'test': test_losses}
    accs = {'test': test_accs}
    logger.success("Done!")
    return sentiment_mix_up, losses, accs
