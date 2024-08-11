import os
import streamlit as st
from fpdf import FPDF

# Import necessary langchain modules
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

# Ensure directories exist
for directory in ['pdfFiles', 'vectorDB', 'generated_pdfs']:
    os.makedirs(directory, exist_ok=True)

# Initialize session state variables
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question",
    )

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(
        persist_directory='vectorDB',
        embedding_function=OllamaEmbeddings(base_url='http://localhost:11434', model="llama3.1")
    )

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(
        base_url="http://localhost:11434",
        model="llama3.1",
        num_predict=1000,
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# App title
st.title("DisasterBot By CodeCommandos - friend when you need the most")

# PDF file uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# Sidebar for downloading generated PDFs
st.sidebar.title("Download Generated PDFs")
generated_files = os.listdir('generated_pdfs')
for file in generated_files:
    file_path = os.path.join('generated_pdfs', file)
    with open(file_path, "rb") as f:
        st.sidebar.download_button(label=f"Download {file}", data=f, file_name=file)

# Process the uploaded PDF and initialize the retriever
if uploaded_file is not None:
    st.text("File uploaded successfully")

    # Save and load PDF content
    pdf_path = f'pdfFiles/{uploaded_file.name}'
    if not os.path.exists(pdf_path):
        with st.spinner("Processing the PDF..."):
            bytes_data = uploaded_file.read()
            with open(pdf_path, 'wb') as f:
                f.write(bytes_data)

            loader = PyPDFLoader(pdf_path)
            data = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )
            all_splits = text_splitter.split_documents(data)

            st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=OllamaEmbeddings(model="llama3.1")
            )
            st.session_state.vectorstore.persist()

    # Initialize the retriever
    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

    # Chat input and response handling
    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(user_input)

        # Placeholder for the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
                formatted_response = f"**Assistant:** {response['result'].strip()}"
                st.markdown(formatted_response)
                chatbot_message = {"role": "assistant", "message": formatted_response}
                st.session_state.chat_history.append(chatbot_message)

                # Generate and save the response as a PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, response['result'])
                pdf_file_path = f"generated_pdfs/{uploaded_file.name}_guidance.pdf"
                pdf.output(pdf_file_path)

                st.success(f"Guidance PDF generated and saved as {pdf_file_path}.")
else:
    st.write("Please upload a PDF file to start the chatbot")
