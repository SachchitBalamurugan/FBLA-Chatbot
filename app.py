from flask import Flask, request, render_template, jsonify
import os
import pdfplumber
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI

# Initialize Flask app
app = Flask(__name__)

# Set your OpenAI API key

os.environ["OPENAI_API_KEY"] = "sk-BwG84-ynNtFs-bbMdF4Ar5sQwpqH_I9VRY3tiFWnbAT3BlbkFJ-5RmmWUGuI-wRjnWH0W_00FV3CvI7ZPdsvDrQhJ-4A"



# Define a function to load PDF and create chunks
def load_pdf_and_chunk(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(text):
        return len(tokenizer.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=24,
        length_function=count_tokens
    )

    chunks = text_splitter.create_documents([text])
    return chunks

# Load PDF data and create vector database
chunks = load_pdf_and_chunk("./fbladata.pdf")
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

# Initialize conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

# Define route for homepage
@app.route("/")
def index():
    return render_template("index.html")

# Define route for chatbot API
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    chat_history = request.json.get("history", [])

    if user_input.lower() == 'exit':
        return jsonify({"answer": "Thank you for using the FBLA chatbot!", "history": chat_history})

    result = qa({"question": user_input, "chat_history": chat_history})
    chat_history.append((user_input, result['answer']))

    return jsonify({"answer": result['answer'], "history": chat_history})

if __name__ == "__main__":
    app.run(debug=True)
