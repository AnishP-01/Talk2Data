!pip install -q transformers datasets sentencepiece

from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import json
import torch

# Step 1: Load Dataset
with open("/content/final_cleaned_sql_dataset_10000.json", "r") as f:
    data = json.load(f)

# Step 2: Format for T5
formatted = [{
    "input_text": f"schema: {item['input']} | question: {item['instruction']}",
    "target_text": item['output']
} for item in data]

dataset = Dataset.from_list(formatted)

# Step 3: Load Model and Tokenizer
model_name = "mrm8488/t5-small-finetuned-wikiSQL"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Step 4: Tokenize Dataset
def preprocess(example):
    tokenized = tokenizer(
        example["input_text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )
    tokenized["labels"] = tokenizer(
        example["target_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )["input_ids"]
    return tokenized

tokenized_dataset = dataset.map(preprocess)

# Step 5: Training Arguments
training_args = TrainingArguments(
    output_dir="./t5-sql",
    per_device_train_batch_size=32,
    num_train_epochs=5,
    learning_rate=3e-4,
    logging_dir="./logs",
    logging_steps=1000,
    save_steps=1000,
    fp16=torch.cuda.is_available()
)

# Step 6: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model)
)

# Step 7: Train
trainer.train()

# Save
model.save_pretrained("t5-sql-finetuned")
tokenizer.save_pretrained("t5-sql-finetuned")



# for testing the model

def generate_sql(question, schema):
    input_text = f"schema: {schema} | question: {question}"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=128)

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example
schema = "employees (serial_no, empoyeid, empoyename, phone_no, email_address, department, position, date_of_joining, attendance, salary, performance_review)"
question = "list of employee salary less than 50000'"
print("Generated SQL:", generate_sql(question, schema))






import shutil
# Zip the trained model directory
shutil.make_archive("t5-sql-finetuned", 'zip', "t5-sql-finetuned")


from google.colab import files
files.download("t5-sql-finetuned.zip")


!pip install -q huggingface_hub

from huggingface_hub import login

# Replace with your token
login(token="Replace with your hugging face token")


from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("t5-sql-finetuned")
tokenizer = T5Tokenizer.from_pretrained("t5-sql-finetuned")

# Upload to your HF repo
model.push_to_hub("t5-small-sql-bot")
tokenizer.push_to_hub("t5-small-sql-bot")
