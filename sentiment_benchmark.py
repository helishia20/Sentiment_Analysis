import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import os
import gc
import pandas as pd
import gradio as gr

# 1-Hardware configuration & setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2-Data Loading Function
def load_data(text_path, label_path):
    if not os.path.exists(text_path) or not os.path.exists(label_path):
        raise FileNotFoundError("please upload train_txt.txt and train_labels.txt")

    with open(text_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    return texts, labels

# -Load the dataset files
texts, labels = load_data('train_text.txt', 'train_labels.txt')

# 3- Dataset Splitting (Train: 80%, Dev: 10%, Test: 10%)
train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
dev_texts, test_texts, dev_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

print(f"Train size: {len(train_texts)} | Dev size: {len(dev_texts)} | Test size: {len(test_texts)}")

# 5-Evaluation Metrics Calculation Function
class TweetDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 6- compute metric prediction
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    #  adding zero_division=0for clear log
    #zero_division=0 prevents redundant logging warnings from Scikit-Learn when m/0 occurs
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 7- main function for testing each model
def run_experiment(model_name, lr=2e-5, epochs=3, max_length=128):
    print(f"\n" + "="*50)
    print(f"شروع آزمایش با مدل: {model_name}")
    print(f"="*50)

    # managing DeBERTa Gradient collapse
    use_fp16 = False if "deberta" in model_name else True


    if "bertweet" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, normalization=True)
    elif "deberta" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    # runnig model with 3 classes
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        ignore_mismatched_sizes=True
    )

    # Tokenization
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    dev_encodings = tokenizer(dev_texts, truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    # Construct PyTorch Dataset Objects
    train_dataset = TweetDataset(train_encodings, train_labels)
    dev_dataset = TweetDataset(dev_encodings, dev_labels)
    test_dataset = TweetDataset(test_encodings, test_labels)

    #Configure Trainer Hyperparameters
    training_args = TrainingArguments(
        output_dir=f'./results_{model_name.split("/")[-1]}',
        num_train_epochs=epochs,
        per_device_train_batch_size=8, # Set to 8 to maintain GPU VRAM stability and avoid OOM crashes
        per_device_eval_batch_size=8,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="no",  # Configured to 'no' to prevent Google Colab Disk Full errors during long runs
        load_best_model_at_end=False, #
        logging_steps=100,
        fp16=use_fp16,
        report_to="none"
    )

    # Initialize Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
    )

    # start training
    trainer.train()

    # final evaluation loop on Dev and Test splits
    print(f"\n--- نتایج ارزیابی مدل {model_name} ---")
    dev_results = trainer.evaluate(eval_dataset=dev_dataset)
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    return dev_results, test_results

# 8-using model and managing the RAM
models_list = [
    'bert-base-cased',
    'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'distilbert/distilbert-base-uncased-finetuned-sst-2-english',
    'albert/albert-base-v2',
    'vinai/bertweet-base',
    'microsoft/deberta-v3-base',
    'google/electra-base-discriminator'
]

all_results = {}

for model_path in models_list:
    try:
        dev_res, test_res = run_experiment(model_path, lr=2e-5, epochs=3)
        all_results[model_path] = {'Dev': dev_res, 'Test': test_res}

        # hardware garbage collection to prevent VRAM memory leaks between model switches
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"خطا در اجرای مدل {model_path}: {e}")

# 9-Render Final Summary Benchmark Report Table
print("\n" + "#"*50 + "\nخلاصه عملکرد مدل‌ها برای جدول گزارش:\n" + "#"*50)
for model_name, res in all_results.items():
    print(f"\nمدل: {model_name}")
    print(f"   [Dev]  Accuracy: {res['Dev']['eval_accuracy']:.4f} | F1: {res['Dev']['eval_f1']:.4f} | Precision: {res['Dev']['eval_precision']:.4f} | Recall: {res['Dev']['eval_recall']:.4f}")
    print(f"   [Test] Accuracy: {res['Test']['eval_accuracy']:.4f} | F1: {res['Test']['eval_f1']:.4f} | Precision: {res['Test']['eval_precision']:.4f} | Recall: {res['Test']['eval_recall']:.4f}")





# ==========================================
#  Model Failure Analysis (Error Analysis)
# ==========================================
import pandas as pd
import torch

# Setting the champion model from our benchmark
best_model_path = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

print(f"==================================================")
print(f"Loading champion model and tokenizer: {best_model_path}")
print(f"==================================================")

# Loading the correct matched tokenizer and model architecture
tokenizer = AutoTokenizer.from_pretrained(best_model_path)
model = AutoModelForSequenceClassification.from_pretrained(best_model_path, num_labels=3).to(device)

model.eval()

# Lists to store failure examples
failed_indices = []
failed_tweets = []
real_labels = []
predicted_labels = []

print("Analyzing test set to extract model misclassifications...")

# Iterating through the test dataset to capture errors
for idx, item in enumerate(test_dataset):
    tweet_text = item['text']
    true_label = item['label']

    # Tokenize text using the RoBERTa tokenizer
    inputs = tokenizer(tweet_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get the class with the highest probability
    pred_label = torch.argmax(outputs.logits, dim=-1).item()

    # If the model's prediction is wrong, save it
    if pred_label != true_label:
        failed_indices.append(idx)
        failed_tweets.append(tweet_text)
        real_labels.append(true_label)
        predicted_labels.append(pred_label)

    # Stop once we collect enough samples for our analysis report
    if len(failed_indices) >= 10:
        break

# Creating the final clean English DataFrame for presentation
error_report_df = pd.DataFrame({
    'Test Index': failed_indices,
    'Tweet Content': failed_tweets,
    'Real Label': real_labels,
    'Predicted Label': predicted_labels
})

print(f"\nSuccessfully extracted {len(error_report_df)} failure examples for the report.")
error_report_df

# ==========================================
# -------------------gradio-----------------
# ==========================================
print("\n" + "="*50 + "\nLaunching Gradio Interactive Dashboard....\n" + "="*50)

def predict_sentiment(tweet_text):
    inputs = tokenizer(tweet_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).flatten().tolist()
    return {
        "🔴 Negative": probs[0],
        "🟡 Neutral": probs[1],
        "🟢 Positive": probs[2]
    }

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Type an English tweet here (e.g., I love this mood!)..."),
    outputs=gr.Label(num_top_classes=3, label="Sentiment Analysis Result"),
    title="Twitter Sentiment Analysis Dashboard",
    description="َThis live web application predicts the sentiment of your text using the fine-tuned Twitter-RoBERTa model.",
    examples=[
        ["I am incredibly happy with the results of this model! :D"],
        ["The weather is average today, nothing special."],
        ["Worst experience ever. The system keeps crashing and failing."]
    ]
)

# shareable link
demo.launch(share=True)