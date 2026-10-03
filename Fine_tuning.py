import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

df=pd.read_csv("dataset.csv")
print(df)
print("Total samples :",len(df))

X = df['text']
Y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y, 
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

train_dataset = Dataset.from_pandas(pd.DataFrame({'text': X_train, 'label': y_train}))
test_dataset = Dataset.from_pandas(pd.DataFrame({'text': X_test, 'label': y_test}))

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(
    tokenize_function, 
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function, 
    batched=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("\nStarting fine-tuning...")

trainer.train()

results = trainer.evaluate()

print("\n--------Evaluation Results--------")
print(results)

trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("\nFine-tuned model saved successfully!")

# Dataset:
# We use a small SMS dataset with two types of messages:
# 0 = Normal message
# 1 = Spam message
# The dataset has more than 20 messages.
#
# Preprocessing:
# 1. Read the dataset using Pandas.
# 2. Separate messages and labels.
# 3. Split the data into training and testing data.
# 4. Convert text into tokens using the DistilBERT tokenizer.
# 5. Make all texts the same length by padding or cutting them.
# 6. Use the processed data to train the model.
