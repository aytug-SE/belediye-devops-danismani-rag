import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

print("1. Belediye dokümanları yükleniyor...")
with open("belediye_veri.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Divide the text into paragraphs
docs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

# Create a FAISS vector store from the documents
print("2. FAISS vektör deposu oluşturuluyor...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Search engine of RAG system.
vector_store = FAISS.from_texts(docs, embeddings) 
print("Vektör deposu hazır!")

print("\n3. Eğitilmiş Fine-tuned LLM modeli yükleniyor...")
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, "./fine_tuned_devops_model")
print("Model yüklendi ve hazır!")

# --- RAG Question Answering ---
def belediye_devops_sor(soru):
    print(f"\n[SORU]: {soru}")

    # Retrieve relevant documents from the FAISS vector store
    related_docs = vector_store.similarity_search(soru, k=1)
    context = related_docs[0].page_content if related_docs else ""

    print(f"\n[BELEDİYE BİLGİ BANKASINDAN ÇEKİLEN BAĞLAM]: \n{context}")

    prompt = f"""<|system|>
You are an expert DevOps Assistant for Municipal IT Infrastructure. Answer the question using the provided Municipal Context. 
<|user|>
Municipal Context: 
{context}

Question: 
{soru}
<|assistant|>
"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.3)

    cevap = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_response = cevap.split("<|assistant|>")[-1].strip()
    return final_response

# --- İLK CANLI TEST (Fonksiyonun Dışında) ---
test_sorusu = "E-Belediye portalında 502 Bad Gateway hatası alıyoruz, hangi IP'li sunucuya bakmalıyım ve servis restart komutu nedir?"
yanit = belediye_devops_sor(test_sorusu)

print("\n" + "=" * 60)
print("[Modelin verdiği yanıt]:")
print(yanit)
print("=" * 60)