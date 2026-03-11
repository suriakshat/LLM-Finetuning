import os
import shutil
import json
import torch
from datetime import datetime
from datasets import Dataset ,concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
# -- 1) ENVIRONMENT & PATHS ---------------------------------------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"
os.environ["HF_TOKEN"] = "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"
MODEL_PATH   = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH    = "/home/ubuntu/collm/ravi_chand_data/data_25_aug.jsonl"
timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR   = "../create_update_and_get_dynamic_25_aug"
OFFLOAD_DIR  = "../offloadcreate_create_update_and_get_dynamic_25_aug"
os.makedirs(OFFLOAD_DIR, exist_ok=True)



print("=-=-=-=-==-=DATASET NAME =-=-=-=--",DATA_PATH)
print("                                           ")
print("=-=-=-=-==-=MODEL NAME =-=-=-=--",MODEL_PATH)

# -- 2) LOAD & FORMAT DATA ----------------------------------------------------------
# raw_data = []
# with open(DATA_PATH, 'r') as f:
#     for i, line in enumerate(f, 1):
#         try:
#             # Replace common control characters that break JSON decoding
#             clean_line = line.replace('\x00', '')  # null byte
#             raw_data.append(json.loads(clean_line))
#         except json.JSONDecodeError as e:
#             print(f"[!] Skipping line {i}: JSONDecodeError: {e}")
 
 
# print("?? Sample raw entry:", raw_data[0])
# # Convert prompt + completion into a single training string
# def convert_to_text(example):
#     return {
#         "text": example["prompt"].strip() + " " + example["completion"].strip()
#     }
#raw_data = []
#with open(DATA_PATH, 'r') as f:
 #   for line in f:
 #       raw_data.append(json.loads(line))
 
#print("?? Sample raw entry:", raw_data[0])


bad_lines = []
raw_data = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):  # i = line number
        line = line.strip()
        if not line:  # skip empty lines
            continue
        try:
            raw_data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"? JSON error at line {i}: {e}")
            print(f"Line content: {line[:200]}")  # show first 200 chars
            bad_lines.append(i)

print(f"\nTotal bad lines: {len(bad_lines)} -> {bad_lines}")

 
print("?? Sample raw entry:", raw_data[0])

print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-===-")

print("?? Last raw entry:", raw_data[-1])



# Convert prompt + completion into a single training string
def convert_to_text(example):
    return {
        "text": example["prompt"].strip() + " " + example["completion"].strip()
    }
dataset = Dataset.from_list(raw_data)
formatted_dataset = dataset.map(convert_to_text)
#formatted_dataset * 4  # Optional: duplicate for reinforcement
#formatted_dataset = concatenate_datasets([formatted_dataset] * 4)
formatted_dataset = Dataset.from_list(formatted_dataset)
 
print("==================================lent=====",len(formatted_dataset))
print("? Sample formatted input:\n", formatted_dataset[0])
# -- 3) TOKENIZER -------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# -- 4) QLoRA CONFIG ----------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
)
# -- 5) LOAD MODEL ------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder=OFFLOAD_DIR,
    offload_state_dict=True,
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
model.resize_token_embeddings(len(tokenizer))
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
# -- 6) LORA ADAPTER CONFIG ---------------------------------------------------------
peft_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# -- 7) TOKENIZATION ---------------------------------------------------------------
def tokenize_fn(batch):
    enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
    enc["labels"] = enc["input_ids"].copy()
    return enc
tok_ds = formatted_dataset.map(tokenize_fn, batched=True, remove_columns=formatted_dataset.column_names)
tok_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# -- 8) COLLATOR -------------------------------------------------------------------
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# -- 9) TRAINING ARGS --------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    num_train_epochs=15,
    logging_steps=100,
    save_strategy="epoch",
    save_total_limit=2,
    no_cuda=False,
    fp16=False,
    bf16=True,
    report_to="none",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    load_best_model_at_end=False
)
# -- 10) TRAIN ----------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok_ds,
    data_collator=collator
)
trainer.train()
# -- 11) SAVE & CLEANUP -------------------------------------------------------------
model.save_pretrained(OUTPUT_DIR)
# trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
shutil.rmtree(OFFLOAD_DIR)
print("save model Dir =========",OUTPUT_DIR)
print("? Training complete and model saved.")