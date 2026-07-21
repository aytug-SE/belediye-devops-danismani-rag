import torch
from transformers import AutoModelForCausalLM, LlamaTokenizerFast, BitsAndBytesConfig 
from peft import LoraConfig, get_peft_model

def load_model_and_tokenizer(model_path):
    print("The model and tokenizer are being loaded from the specified path...")
    
    tokenizer = LlamaTokenizerFast.from_pretrained(
        model_path, 
        local_files_only=True, 
        legacy=False
    )
    
    # Because of the tokenizer's behavior, we need to set the pad_token to eos_token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quantity_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )    

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True
    )
    
    # Configuration of LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    print("The model is being made PEFT-compatible.")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer