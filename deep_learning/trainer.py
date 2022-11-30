import torch
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

import utils

def train(train_dataloader, model, criterion, optimizer, writer, epoch, device):
    model.train()
    for idx, (label, text, offsets) in enumerate(train_dataloader):
        label, text, offsets = label.to(device), text.to(device), offsets.to(device)
        preds = model(text, offsets)
        loss = criterion(preds, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        optimizer.zero_grad()
        if idx%250 == 0:
            with torch.no_grad():
                metrics = calculate_metrics(label.detach().cpu().numpy(), (preds.detach().cpu().numpy()>0.5)*1)
                writer.add_scalar('Train Batch Loss', loss, epoch*len(train_dataloader) + idx)
                writer.add_scalar('Train Batch Accuracy', metrics["accuracy"], epoch*len(train_dataloader) + idx)
                writer.add_scalar('Train Batch Micro Average F1', metrics["micro_f1"], epoch*len(train_dataloader) + idx)
                writer.add_scalar('Train Batch Macro Average F1', metrics["macro_f1"], epoch*len(train_dataloader) + idx)
        

def evaluate(val_dataloader, model, criterion, writer, epoch, device):
    model.eval()
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(val_dataloader):
            label, text, offsets = label.to(device), text.to(device), offsets.to(device)
            preds = model(text, offsets)
            loss = criterion(preds, label)
            metrics = calculate_metrics(label.detach().cpu().numpy(), (preds.detach().cpu().numpy()>0.5)*1)
            writer.add_scalar('Val Accuracy', metrics["accuracy"], epoch*len(val_dataloader) + idx)
            writer.add_text(f"Val Metrics", utils.pretty_json(metrics["report"]), epoch)
            writer.add_scalar('Val Loss', loss, epoch*len(val_dataloader) + idx)
    

def calculate_metrics(true, preds):
    report = classification_report(true, preds, output_dict=True, zero_division=1, target_names=['toxic','severe_toxic','obscene','threat','insult','identity_hate'])
    results = {
        "report": report,
        "accuracy": accuracy_score(true, preds),
        "macro_r": report['macro avg']["recall"],
        "macro_p": report['macro avg']["precision"],
        "macro_f1": report['macro avg']["f1-score"],
        "micro_r": report['micro avg']["recall"],
        "micro_p": report['micro avg']["precision"],
        "micro_f1": report['micro avg']["f1-score"]
    }
    return results
