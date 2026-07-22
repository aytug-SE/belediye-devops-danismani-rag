from datasets import load_dataset

def get_prepared_dataset(file_path):
    print("Dataset is loading...")
    # Load the dataset from the specified file path using the Hugging Face 'load_dataset' function
    dataset = load_dataset("json", data_files=file_path, split="train")
    print(f"Dataset successfully loaded. Total number of rows.: {len(dataset)}")
    return dataset

def formatting_prompts_func(example):
    instruction = example.get("instruction", "")
    user_input = example.get("input", "")
    output = example.get("output", "")

    # TinyLlama için ideal chat şablonu (Instruction ve Input ayrılmış durumda)
    text = (
        f"<|system|>\n"
        f"You are a helpful DevOps Assistant\n"
        f"<|user|>\n"
        f"{instruction}\n\nLog/System Context:\n{user_input}\n"
        f"<|assistant|>\n"
        f"{output}"
    )


    return text
