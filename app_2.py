import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
# from langchain.chat_models import ChatGPT2  # Change to ChatGPT2
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import gpt_2_simple as gpt2
from transformers import GPT2Tokenizer, GPT2Model
import torch

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1002,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# def get_vectorstore(text_chunks):
#     embeddings = OpenAIEmbeddings()
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore
def get_embeddings(text_chunks):
    # Initialize GPT-2 model
    model_name = "gpt2"  # You can adjust the model name based on your needs
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Add a new pad token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    model = GPT2Model.from_pretrained(model_name)

    # Tokenize and obtain embeddings
    inputs = tokenizer(text_chunks, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract embeddings from the last hidden states
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()

    return embeddings

def get_conversation_chain(text_chunks):
    # Get embeddings using GPT-2 and Hugging Face Transformers
    embeddings = get_embeddings(text_chunks)

    # Assuming FAISS is used as a vector retriever
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    # Initialize the GPT-2 model for language modeling
    gpt2_model = gpt2.start_tf_sess()
    gpt2.load_gpt2(gpt2_model)

    llm = {
        "model": gpt2_model,
        "tokenizer": None,  # Adjust if a tokenizer is required
        # Add other parameters based on the requirements of ConversationalRetrievalChain.from_llm
    }

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore  # Provide the vector retriever here
    )
    return conversation_chain
# def get_conversation_chain(text_chunks):
#     # Initialize the GPT-2 model
#     gpt2_model = gpt2.start_tf_sess()
#     gpt2.load_gpt2(gpt2_model)
    
#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=gpt2,  # Use GPT-2 as the language model
#         memory=memory
#     )
#     return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                # vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    text_chunks)

if __name__ == '__main__':
    main()
