import os
import re
import ast
import pymupdf
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset


MODEL_NAME = "gpt2"

# Asking Q!

prompt = input('He is Risen. President Nixon is here to lend an ear to your request.\n')

# --------------------- COMPILING DATA --------------------- #

# dialogue_files = []
# directory = 'trainingdata/'

# Initializing PDF reader, looping through local directory, then going page-by-page
# through each pdf and scraping relevant dialogue

# for file in os.listdir(directory):
#     path = os.path.join(directory, file)
#     doc = pymupdf.open(path)
#     for page in doc:
#         nix_pat = r"((?:PRESIDENT|RN):\s*(?:[^\n]+(?:\n(?![A-Z]+:))?)*)"
#
#         page_text = page.get_text()
#         nixonspeech = re.findall(nix_pat, page_text)
#         dialogue_files.append(nixonspeech)
#
# # print(dialogue_files)


# Writing relevant dialogue to a text file for gpt-2 feeding

# with open('dialogue.txt', 'w') as file:
#     for i in dialogue_files:
#         i = str(i)
#         file.write(i + '\n')

# ------------------------- CHATBOT INITIALIZING ------------------------- #

# Initializing model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


# Getting data from .txt and tokenizing for training
with open('dialogue.txt', 'r') as f:
    raw_lines = f.readlines()

# Process the dialogues
nixon_dialogues = []
for line in raw_lines:
    try:
        # Safely evaluate the string as a list
        dialogue_list = ast.literal_eval(line.strip())

        # Process each dialogue in the list
        for dialogue in dialogue_list:
            # Clean up the text
            cleaned = dialogue.replace("PRESIDENT:\n", "")
            cleaned = cleaned.replace("\n", " ")
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Replace multiple spaces with single space
            cleaned = cleaned.strip()

            nixon_dialogues.append(cleaned)
    except:
        print(f"Skipping: {line}")  # Skip any malformed lines

# Create initial dataset from dialogues
raw_dataset = Dataset.from_dict({"text": nixon_dialogues})


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )


tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=raw_dataset.column_names
)

# Compiling training data
training_args = TrainingArguments(
    output_dir='./NixonBot',
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    warmup_steps=100,
    save_steps=500,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Actually training the model
trainer.train()

# Saving the model
trainer.save_model("./NixonBot")


# Generating response
def generate_response(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(
        inputs['input_ids'],
        max_length=200,
        top_p=0.92,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penality=1.2
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)


generate_response(prompt)
