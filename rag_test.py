import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

print("Municipality documents are loading...")
with open("belediye_veri.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Divide the text into paragraphs
docs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

# Create a FAISS vector store from the documents
print("The FAISS Vector store is being created...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Search engine of RAG system.
vector_store = FAISS.from_texts(docs, embeddings) 
print("The vector store is ready.")

print("\nTrained Fine-Tuned LLM model is being loaded...")
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "./fine_tuned_devops_model")
print("The model is loaded and prepared!!!")

# --- RAG Question Answering ---
def municipality_devops_ask(question):
    print(f"\n[SORU]: {question}")

    # Retrieve relevant documents from the FAISS vector store
    related_docs = vector_store.similarity_search(question, k=1)
    context = related_docs[0].page_content if related_docs else ""

    print(f"\n[CONTEXT DRAWN FROM THE MUNICIPALITY INFORMATION BANK]: \n{context}")

    prompt = f"""<|system|>
You are an expert DevOps Assistant for Municipal IT Infrastructure. Answer the question accurately using the provided Municipal Context.
<|user|>
Municipal Context: 
{context}

Question: 
{question}
<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.3)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_response = response.split("<|assistant|>")[-1].strip()
    return final_response
