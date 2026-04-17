# ======================================
# IMPORT LIBRARIES
# ======================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from tqdm import tqdm

# ======================================
# LOAD DATA
# ======================================
df = pd.read_csv("rating.csv")
df = df.head(10000)

print("Dataset Shape:", df.shape)

# ======================================
# TARGET & FEATURE
# ======================================
target_column = "title_sentiment"

# Auto-detect text column (excluding target)
text_columns = df.select_dtypes(include=['object']).columns.tolist()
text_columns.remove(target_column)
text_column = text_columns[0]

print("Selected Text Column:", text_column)

X_text = df[text_column].astype(str)
y = df[target_column]

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# ======================================
# TRAIN TEST SPLIT
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# ======================================
# TOKENIZER
# ======================================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_texts(texts):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

# Encode train data
train_encodings = encode_texts(X_train)
test_encodings = encode_texts(X_test)

# ======================================
# APPLY SMOTE ON INPUT IDS
# ======================================
smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(
    train_encodings['input_ids'].numpy(), y_train
)

X_resampled = torch.tensor(X_resampled)
y_resampled = torch.tensor(y_resampled)

# ======================================
# DATASET CLASS
# ======================================
class NewsDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.input_ids = input_ids
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": (self.input_ids[idx] != 0).long(),
            "labels": self.labels[idx]
        }

train_dataset = NewsDataset(X_resampled, y_resampled)
test_dataset = NewsDataset(test_encodings['input_ids'], torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ======================================
# BERT MODEL
# ======================================
class BERTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        cls_output = outputs.pooler_output
        x = self.dropout(cls_output)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BERTClassifier(num_classes=len(np.unique(y)))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ======================================
# TRAINING LOOP
# ======================================
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch+1}/{epochs}")

    for batch in tqdm(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Training Loss: {total_loss:.4f}")
    print(f"Training Accuracy: {train_acc:.4f}")

# ======================================
# EVALUATION
# ======================================
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)

        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(true_labels, predictions)
cm = confusion_matrix(true_labels, predictions)

print("\nTest Accuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n",
      classification_report(true_labels, predictions))
