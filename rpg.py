import os
from sys import prefix

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from numpy import dot, array
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import yaml
import document as doc
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import streamlit as st

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
os.environ['PINECONE_API_KEY'] = config['PINECONE_API_KEY']


st.title('Me lembre...')

document = doc.read_markdown_file('data/Caminho.md')


all_documents = [document]


# Dividir documentos em chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=5500,
    chunk_overlap=500,
    length_function=len
)
chunks = text_splitter.create_documents(all_documents)

print("\nChunks gerados:")
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}: {chunk.page_content}")


embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
# print(embeddings)

embedded_chunks = embeddings.embed_documents([chunk.page_content for chunk in chunks])

# Mostrar os embeddings gerados
# print("\nEmbeddings gerados (mostrando apenas os primeiros 5 elementos de cada):")
# for i, embed in enumerate(embedded_chunks):
#     print(f"Embedding {i+1}: {embed[:5]}...")

query = st.text_area('Digite sua pergunta:')

index_name = 'rpg'
vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)


llm = ChatOpenAI(model='gpt-4', temperature=0.2)
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
# answer_1 = chain.invoke(query_1)
prefix_query = 'Responda apenas com base no input fornecido. '

print('Tudo pronto! Agora você pode fazer perguntas ao modelo. ')


if st.button('Consultar'):
    print('consulta')
    question = prefix_query + query
    print(question)
    answer = chain.invoke(question)
    print(answer['result'])
    st.write(answer['result'])