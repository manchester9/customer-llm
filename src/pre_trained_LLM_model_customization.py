from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, TrainingArguments, Trainer
import torch
import openai 

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Prepare training data
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="path_to_your_train_data.txt",
    block_size=128,
)

# Prepare validation data
validation_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="path_to_your_validation_data.txt",
    block_size=128,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
)

# Create the Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=lambda data: {"input_ids": torch.stack([f[0] for f in data])},
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

# Train the model
trainer.train()


######## different task ########
###############################
# Instead of generating text classify text by adding a classification head using a classification loss function, and training-evaluating it on labeled data
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

# Add a classification head to the model
model.classifier = torch.nn.Linear(model.config.n_embd, 2)

# Prepare data
class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Load your data and split it into training and validation sets
data = pd.read_csv('path_to_your_data.csv')
train_texts, val_texts, train_labels, val_labels = train_test_split(data.text, data.label, test_size=0.2)

# Create data loaders
train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, 128)
val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, 128)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Train the model
optimizer = AdamW(model.parameters(), lr=1e-5)

for epoch in range(10):  # Number of epochs
    # Train
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = model.classifier(outputs.last_hidden_state[:, 0, :])
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

    # Validate
    model.eval()
    total = correct = 0
    for batch in val_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = model.classifier(outputs.last_hidden_state[:, 0, :])
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch: {epoch + 1}, Validation Accuracy: {accuracy:.4f}')


