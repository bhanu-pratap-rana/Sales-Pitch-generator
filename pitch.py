import streamlit as st
import os
import tempfile
import warnings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import time

# Filter torch warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize Streamlit app
st.title("Sales Pitch Generator from PDF Content")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'vectors' not in st.session_state:
    st.session_state.vectors = None

# Hardcoded API key and model
GROQ_API_KEY = "YOUR_API_KEY"
MODEL_NAME = "mixtral-8x7b-32768"  # Updated to a valid Groq model

# Initialize the Groq language model
@st.cache_resource
def get_llm():
    try:
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=MODEL_NAME,
            temperature=0.7,
            max_tokens=1000
        )
    except Exception as e:
        st.error(f"Error initializing Groq model: {str(e)}")
        return None

# Initialize embeddings model
@st.cache_resource
def get_embeddings():
    try:
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            cache_folder=os.path.join(os.getcwd(), "models")
        )
    except Exception as e:
        st.error(f"Error initializing embeddings model: {str(e)}")
        return None

# Create a prompt template for generating a sales pitch
prompt = ChatPromptTemplate.from_template("""
    Generate a compelling sales pitch based on the provided context from the document.
    Make sure to highlight the most relevant points and tailor the pitch to be persuasive.
    
    Key points to focus on: {input}
    
    Context from document:
    {context}
    
    Sales Pitch:
""")

# Function to create vector embeddings from PDF documents
def vector_embedding(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        final_documents = text_splitter.split_documents(docs)
        texts = [doc.page_content for doc in final_documents]
        
        embeddings = get_embeddings()
        if embeddings is None:
            return False
            
        # Store in session state
        st.session_state.documents = final_documents
        st.session_state.vectors = FAISS.from_texts(
            texts,
            embeddings
        )
        return True
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return False

# Main app interface
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
prompt_input = st.text_input("Enter the key points you want to focus on in the sales pitch")

if st.button("Generate Document Embeddings") and uploaded_file:
    with st.spinner("Processing PDF..."):
        try:
            # Create a temporary file with context manager
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Process the file
            success = vector_embedding(tmp_file_path)
            
            # Clean up the temporary file
            try:
                tmp_file.close()
                os.unlink(tmp_file_path)
            except Exception as e:
                st.warning(f"Note: Temporary file cleanup failed: {str(e)}")
            
            if success:
                st.success("Documents are loaded and ready for processing")
                
        except Exception as e:
            st.error(f"Error during file processing: {str(e)}")

if prompt_input and st.session_state.vectors is not None:
    with st.spinner(f"Generating sales pitch using {MODEL_NAME}..."):
        try:
            llm = get_llm()
            if llm is None:
                st.error("Failed to initialize the Groq model. Please try again.")
                st.stop()
                
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            start_time = time.time()
            response = retrieval_chain.invoke({'input': prompt_input})
            end_time = time.time()
            
            st.write("Generated Sales Pitch:")
            st.write(response['answer'])
            st.write(f"Response time: {end_time - start_time:.2f} seconds")
            
            with st.expander("Document Similarity Search Results"):
                for i, doc in enumerate(response["context"], 1):
                    st.write(f"Relevant Section {i}:")
                    st.write(doc.page_content)
                    st.divider()
        except Exception as e:
            st.error(f"Error generating sales pitch: {str(e)}")