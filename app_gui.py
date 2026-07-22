import gradio as gr
from rag_test import municipality_devops_ask

def gui_response(message, history):
    response = municipality_devops_ask(message)
    return response

# Prepared Chat Interface
demo = gr.ChatInterface(
    fn=gui_response,
    title="Municipality IT infrastructure DevOps Assistant",
    description="TinyLlama Fine-Tuned + RAG (FAISS) based smarter analyzer system.",
    examples=[
        "Analyze the log and state the root cause with solution steps: E-MUNICIPALITY PORTAL: Nginx error log shows '111: Connection refused while connecting to upstream'. IP: 10.10.1.50. HTTP Status: 502 Bad Gateway.",
        "Analyze the log and state the root cause with solution steps: EDMS/EBYS (Belgenet) Server at 10.10.1.60: Java Tomcat service is frozen and unresponsive. Users report E-Signature verification failure via Kamu SM OCSP servers.",
        "Analyze the log and state the root cause with solution steps: K8S_CLUSTER: Namespace 'kbs-prod'. MapServer pods experience severe memory leak and high latency during GIS map rendering."
    ],
)

if __name__ == "__main__":
    demo.launch(share=False) # Executes on Local host