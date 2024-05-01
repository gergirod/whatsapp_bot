from flask import Flask, request
import os
from twilio.rest import Client
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS


app = Flask(__name__)

load_dotenv()


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
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
chunks = get_text_chunks(get_pdf_text(["", "", "", ""]))
store = get_vectorstore(chunks)
pdf_qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0, model_name='gpt-4'),
    retriever=store.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

chat_history = []

@app.route("/test", methods=["POST"])
def message():
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')
    client = Client(account_sid, auth_token)
    twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')
    sender_phone_number = request.values.get('From', '')
    messageType = request.values.get('MessageType')
    if messageType == 'text':
        user_question = request.values.get('Body')
        result = pdf_qa.invoke({"question": user_question, "chat_history": chat_history})
        chat_history.append((user_question, result["answer"]))
        message = client.messages.create(
                body= result["answer"],
                from_=twilio_phone_number,
                to=sender_phone_number
        )
        return str(message.sid)
    

if __name__ == "__main__":
    app.run(debug=True)