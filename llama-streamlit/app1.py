from flask import Flask, jsonify, request, send_file, render_template, session
from flask_session import Session
import os
import redis
from fpdf import FPDF

# Import necessary langchain modules (simulate if not available)
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

app = Flask(__name__)

# Configure Flask-Session with Redis
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_KEY_PREFIX'] = 'session:'
app.config['SESSION_REDIS'] = redis.StrictRedis(host='localhost', port=6379, db=0)
sess = Session(app)

# Ensure directories exist
for directory in ['pdfFiles', 'vectorDB', 'generated_pdfs']:
    os.makedirs(directory, exist_ok=True)

# Initialize session state variables
session_state = {
    'template': """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:""",
    'prompt': PromptTemplate(
        input_variables=["history", "context", "question"],
        template="""You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

        Context: {context}
        History: {history}

        User: {question}
        Chatbot:"""
    ),
    'memory': ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question",
    ),
    'vectorstore': Chroma(
        persist_directory='vectorDB',
        embedding_function=OllamaEmbeddings(base_url='http://localhost:11434', model="llama3.1")
    ),
    'llm': Ollama(
        base_url="http://localhost:11434",
        model="llama3.1",
        num_predict=1000,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    ),
    'chat_history': [],
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_pdf():
    file = request.files.get('file')
    if file and file.filename.endswith('.pdf'):
        file_path = f'pdfFiles/{file.filename}'
        file.save(file_path)

        # Process the uploaded PDF
        loader = PyPDFLoader(file_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len
        )
        all_splits = text_splitter.split_documents(data)

        session_state['vectorstore'] = Chroma.from_documents(
            documents=all_splits,
            embedding=OllamaEmbeddings(model="llama3.1")
        )
        session_state['vectorstore'].persist()

        return jsonify({"message": "PDF processed successfully."})
    return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('input', '')

    # Initialize the retriever
    retriever = session_state['vectorstore'].as_retriever()

    if 'qa_chain' not in session_state:
        session_state['qa_chain'] = RetrievalQA.from_chain_type(
            llm=session_state['llm'],
            chain_type='stuff',
            retriever=retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": session_state['prompt'],
                "memory": session_state['memory'],
            }
        )

    response = session_state['qa_chain'](user_input)
    formatted_response = response['result'].strip()

    # Generate and save the response as a PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, formatted_response)
    pdf_file_path = f"generated_pdfs/response.pdf"
    pdf.output(pdf_file_path)

    # Update chat history
    session_state['chat_history'].append({"role": "user", "message": user_input})
    session_state['chat_history'].append({"role": "assistant", "message": formatted_response})

    return jsonify({
        "response": formatted_response,
        "pdf_file": "response.pdf"
    })

@app.route('/api/download', methods=['GET'])
def download_pdf():
    filename = request.args.get('filename', 'response.pdf')
    file_path = os.path.join('generated_pdfs', filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(port=5000)
