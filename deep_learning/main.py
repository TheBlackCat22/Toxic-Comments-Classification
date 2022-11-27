from sklearn.model_selection import train_test_split
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import json

import utils
import dataset
import models
import trainer


# Parsing Arguments
print("\nParameters")
print("-----------------")

args = utils.get_args()
print(json.dumps(vars(args), indent=2))


# Importing Data
print("\n\nStarted Importing Data")
print("------------------------")

train_df = utils.data_importer(args.train_data_path, train=True)
test_df = utils.data_importer(args.test_data_path, train=False)


# Train, Val split
print("\n\nSplitting Training Data into Train set and Val set in 4:1 Ratio.")
print("------------------------------------------------------------------")

X_train, X_val, y_train, y_val = train_test_split(
    train_df.iloc[:, :2], train_df.iloc[:, 2:], test_size=0.2, random_state=42, stratify=(train_df["no_toxicity"].values))

print("Train Data")
train_dataset = dataset.Toxic_Comment_Dataset(X_train, y_train)
print("Train Size:", len(train_dataset))
print("\nVal Data")
val_dataset = dataset.Toxic_Comment_Dataset(X_val, y_val)
print("Val Size:", len(val_dataset))


# Creating Vocabulary
print("\n\nCreating Vocabulary")
print("--------------------")

vocab = dataset.get_vocabulary(train_dataset)
print("Vocabulary Size:" ,len(vocab))


# Creating Dataloader
print("\n\nCreating Dataloader")
print("--------------------")

train_loader, val_loader = dataset.get_dataloader(train_dataset, val_dataset, vocab, args.batch_size)
print("Num of Batches in Train Loader: ", len(train_loader))
print("Num of Batches in Val Loader: ", len(val_loader))


# Train Loop
print("\n\nTraining")
print("-----------")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.get_network(args.net)(len(vocab), args.embed_dim, 6)
model = model.to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr= args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma= args.lr_decay)

writer = SummaryWriter()

for epoch in tqdm(range(0, args.num_epochs), desc="Epochs", unit="epoch"):
    trainer.train(train_loader, model, criterion, optimizer, writer, epoch, device)
    trainer.evaluate(val_loader, model, criterion, writer, epoch, device)
    scheduler.step()

writer.close()


print("\nSaving Model")
torch.save(model, args.save_folder+args.net+".pt")

print("\nDONE!!")