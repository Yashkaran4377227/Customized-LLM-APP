# 🕵️‍♂️ CyberSage: AI-Powered Threat Intelligence Advisor with RAG

## 📚 Powered by Expert Knowledge and RAG Technology
CyberSage is an advanced implementation of a Retrieval-Augmented Generation (RAG) bot, drawing its wisdom from "Cyber Threat Intelligence" by Ali Dehghantanha, Mauro Conti, and Tooska Dargahi. This project builds upon the Customized-LLM-APP framework, enhancing it for cybersecurity applications.

---

## 🔍 What is CyberSage?

CyberSage is your AI companion in the realm of cybersecurity threat intelligence. It harnesses the power of RAG to provide more accurate and contextually relevant responses. By incorporating external knowledge, CyberSage offers insights on:

- 🦠 Latest cyber threats
- 🛡️ Emerging vulnerabilities
- 🔧 Effective mitigation strategies

---

## 🚀 Key Features

- **RAG-Enhanced AI Brain**: Utilizes Zephyr-7b-beta with RAG for intelligent threat analysis
- **Dynamic Responses**: Adjustable parameters for customized outputs
- **Sleek Interface**: Smooth chatting experience via Gradio
- **Expert Knowledge Base**: Insights from leading cybersecurity experts, indexed and retrieved using RAG

---

## 🛠️ Tech Stack

- **LLM**: HuggingFaceH4/zephyr-7b-beta
- **Embedding Model**: all-MiniLM-L6-v2 sentence transformer
- **UI**: Gradio
- **Backend**: Python
- **Package Manager**: pip

---

## 🏁 Quick Start Guide

1. **Clone CyberSage**
   ```
   git clone https://github.com/Yashkaran4377227/Customized-LLM-APP/.git
   cd Customized-LLM-APP
   ```

2. **Install Requirements**
   ```
   pip install -r requirements.txt
   ```

3. **Launch CyberSage**
   ```
   python app.py
   ```

4. **Start Chatting**
   Open your browser and navigate to the local URL displayed in your terminal

---

## 🎛️ Customize Your Experience

Tweak these parameters in the Gradio interface:

- **System Message**: Define CyberSage's role and personality
- **Max New Tokens**: Control the length of responses (1-2048)
- **Temperature**: Adjust the creativity level (0.1-4.0)
- **Top-p**: Fine-tune response diversity (0.1-1.0)

---

## 🧠 How RAG Enhances CyberSage

CyberSage uses RAG to improve its performance by:

1. **Indexing**: The cybersecurity knowledge base is indexed into a vector store.
2. **Retrieval**: When a query is received, relevant documents are retrieved from the index.
3. **Generation**: The retrieved information is combined with the original prompt for more informed responses.

This approach allows CyberSage to access up-to-date and domain-specific cybersecurity information without extensive retraining.

---

## 🚨 Important Note

While CyberSage offers valuable insights, it's not a substitute for professional cybersecurity expertise. Always consult certified professionals for critical security decisions.

---

## 🤝 Join the CyberSage Community

We welcome contributors! Here's how you can help:

- 🐛 Spot a bug? Open an issue!
- 💡 Have an idea? Share it!
- 🔧 Want to improve the code? Submit a pull request!
---

## 📞 Get in Touch

Have questions or suggestions? Reach out:
- 📧 Email: ya4377227@alphacollege.me

Stay one step ahead in the cyber world with CyberSage! 🔐🤖
