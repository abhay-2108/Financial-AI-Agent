## 📊 FinAgent – Financial Chatbot

**FinAgent** is an AI-powered Streamlit web application that allows users to **upload financial documents (PDF, Excel, CSV, TXT, HTML)** and chat with them in real-time using LLMs. It leverages document embeddings, retrieval-augmented generation (RAG), and a lightweight chatbot interface to help users extract insights from complex financial files.

---

### 🚀 Features

* 📎 Upload financial files (`.pdf`, `.csv`, `.xlsx`, `.txt`, `.html`)
* 🤖 Ask natural language questions (e.g., "What was the revenue in 2023?")
* 🧠 AI uses document context to answer with high relevance
* 🔍 Retrieval-augmented generation using vector similarity search
* 💬 Chat interface with history display
* ⚡ Powered by **LangChain**, **Ollama**, **ChromaDB**, and **Streamlit**

---

### 📂 File Support

* PDF: Financial reports, statements, and contracts
* Excel: Balance sheets, audit sheets
* CSV: Tabular transaction data, market data
* TXT/HTML: Raw or web-based financial text

---

### 🛠️ Tech Stack

| Component    | Technology                                      |
| ------------ | ----------------------------------------------- |
| LLM          | [Ollama](https://ollama.com) with `llama3.1:8b` |
| Vector Store | ChromaDB                                        |
| Embeddings   | Ollama Embeddings                               |
| UI Framework | Streamlit                                       |
| RAG Pipeline | LangChain                                       |


### 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/finagent-chatbot.git
cd finagent-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama (make sure you have it installed)
ollama run llama3.1:8b

# 4. Run the app
streamlit run app.py
```

---

### 📁 Directory Structure

```
.
├── financial.py                   # Main Streamlit app
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
```

* A logo or banner image
* HuggingFace or Vercel deployment steps
* Docker support instructions

Would you like me to generate a `requirements.txt` file too?
