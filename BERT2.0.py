import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load AGNews dataset
dataset = load_dataset("ag_news")

# Limit the dataset to 2500 samples
train_texts = dataset['train']['text'][:500]  # Limit to first 2500 samples
train_labels = [1 if label == 3 else 0 for label in dataset['train']['label'][:500]]  # Labels for Human vs AI

# Split into train and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)

# Preprocess the data
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long).to(device)
        }

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # Human vs AI
model.to(device)

# Prepare training and validation datasets
train_dataset = TextDataset(train_texts, train_labels, tokenizer)
val_dataset = TextDataset(val_texts, val_labels, tokenizer)

# Define compute_metrics function for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Set training arguments with a progress bar during training
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save model outputs
    evaluation_strategy="epoch",  # Evaluate at the end of every epoch
    learning_rate=2e-5,  # Learning rate for optimizer
    per_device_train_batch_size=8,  # Training batch size per device
    per_device_eval_batch_size=8,  # Evaluation batch size per device
    num_train_epochs=3,  # Number of epochs to train
    weight_decay=0.01,  # Weight decay to prevent overfitting
    logging_dir="./logs",  # Directory to save logs
    logging_steps=10,  # Log every 10 steps
    save_steps=500,  # Save checkpoint every 500 steps
    report_to="none"  # Disable logging to external services (like TensorBoard, etc.)
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model with progress bar
print("Training the model...")
trainer.train()  # This will automatically show a progress bar during training

# Save the model
model.save_pretrained("human_vs_ai_model")
tokenizer.save_pretrained("human_vs_ai_model")

# Load the model back if needed
model = BertForSequenceClassification.from_pretrained("human_vs_ai_model").to(device)
tokenizer = BertTokenizer.from_pretrained("human_vs_ai_model")

# Function to predict if text is human or AI-generated
def predict_text(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    return "AI-generated" if predicted_label == 1 else "Human-written"

# Function to get multiple inputs from the user and predict
def test_model_on_user_inputs():
    print("Enter text for detection (type 'exit' to stop):")
    predictions = []
    
    # Continuously take user input until 'exit' is typed
    while True:
        user_input = input("Enter text: ")
        if user_input.lower() == 'exit':
            break
        
        prediction = predict_text(user_input)
        predictions.append((user_input, prediction))
    
    # Show predictions
    for text, prediction in predictions:
        print(f"Text: {text}\nPrediction: {prediction}\n")

# Run the testing function
test_model_on_user_inputs()
