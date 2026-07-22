# 🏛️ Belediye DevOps & IT Danışmanı (LLM + RAG) (İngilizce)

Bu proje, belediye IT altyapılarında karşılaşılan sunucu, ağ ve sistem arızalarını hızlıca teşhis etmek ve çözüm sunmak amacıyla geliştirilmiş **domain-specific (alana özel)** bir yapay zeka asistanıdır.

## 🛠️ Mimari ve Teknolojiler
- **Base Model:** TinyLlama 1.1B
- **Fine-Tuning:** LoRA (PEFT) & Hugging Face `SFTTrainer`
- **Retrieval-Augmented Generation (RAG):** FAISS Vector Store + Custom Embeddings
- **Dil:** Python 3.10+

## 📁 Proje Yapısı
- `train.py`: TinyLlama modelini belediye DevOps verisetiyle fine-tune eden ana eğitim scripti.
- `rag_test.py`: FAISS tabanlı RAG mimarisi ile fine-tuned modeli birleştiren çıkarım (inference) dosyası.
- `fine_tuned_devops_model/`: LoRA adaptör ağırlıkları (`adapter_model.safetensors` ve `adapter_config.json`).
- `cloud_dataset.jsonl`: Fine-tuning eğitimi için hazırlanan JSONL formatındaki özel DevOps veri seti.
- `belediye_veri.txt`: RAG mimarisinde FAISS vektör veritabanına aktarılan bilgi tabanı (knowledge base).
- `model_config.py` & `dataset_loader.py`: Model ve veri seti konfigürasyon/yükleme yardımcı dosyaları.
- `requirements.txt`: Projenin bağımlılıklarını içeren kütüphane listesi.

## 🚀 Kurulum ve Çalıştırma
```bash
pip install -r requirements.txt
python rag_test.py
