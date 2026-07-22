from transformers import TrainingArguments, Trainer
from trl import SFTTrainer
import model_config as m_cfg
import dataset_loader as d_ld

MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = "cloud_dataset.jsonl"

# Execute the functions to load the model, tokenizer, and dataset
model, tokenizer = m_cfg.load_model_and_tokenizer(MODEL_PATH)
dataset = d_ld.get_prepared_dataset(DATASET_PATH)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=1,
    max_steps=20,
    fp16=False,
    optim="adamw_torch",
    report_to="none"
)

# Build the Trainer engine
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class = tokenizer,
    formatting_func = d_ld.formatting_prompts_func,  # Connect the formatting function to the Trainer
)

print("The training process is starting...")

# Start the training process and save the model
trainer.train()
trainer.model.save_pretrained("./fine_tuned_devops_model")

print("The training process has completed. The fine-tuned model has been saved to './fine_tuned_devops_model'.")
