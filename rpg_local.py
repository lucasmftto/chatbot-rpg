import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings  # Usando embeddings locais do Ollama
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.llms import Ollama  # Usando Ollama como LLM
from langchain.chains import RetrievalQA
import streamlit as st
import yaml
import document as doc

# Carregar configurações ao iniciar o app
with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)
os.environ['PINECONE_API_KEY'] = config['PINECONE_API_KEY']
os.environ['PINECONE_ENV'] = config['PINECONE_ENV']

# Configurações do Ollama
OLLAMA_BASE_URL = "http://localhost:11434"  # URL do Ollama local
OLLAMA_MODEL = "llama2"  # Modelo do Ollama que você deseja usar

index_name = 'rpg-full'
embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)  # Usando embeddings do Ollama

def index_exists(index_name):
    try:
        # Tenta carregar o índice
        PineconeVectorStore.from_existing_index(index_name, embeddings, text_key="text")
        return True
    except Exception as e:
        print(f"Erro ao carregar o índice: {e}")
        return False

def read_all_markdown_files(folder_path):
    all_documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.md'):
            file_path = os.path.join(folder_path, filename)
            document = doc.read_markdown_file(file_path)
            all_documents.append(document)
    return all_documents

# Função para verificar e criar o índice apenas uma vez
@st.cache_resource
def get_or_create_index():
    # Verificar se o índice existe
    if not index_exists(index_name):
        print(f"Criando o índice '{index_name}' no Pinecone...")

        # Carregar o documento uma única vez
        all_documents = read_all_markdown_files('data')

        # Dividir documentos em chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5500,
            chunk_overlap=500,
            length_function=len
        )
        chunks = text_splitter.create_documents(all_documents)

        vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=index_name)

        print("Índice criado e documentos salvos no Pinecone!")
        return vector_store
    else:
        print(f"Usando o índice existente: '{index_name}'")
        return PineconeVectorStore.from_existing_index(index_name, embeddings, text_key='text')

# Carregar o vetor ou criar se não existir
vector_store = get_or_create_index()

# Configurar o modelo LLM e o encadeamento de consulta
llm = Ollama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)  # Usando Ollama como LLM
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

# Prefixo para a consulta
prefix_query = 'Responda apenas com base no input fornecido. '

# Inicialização do título e da área de consulta
st.title('Me lembre...')
query = st.text_area('Digite sua pergunta:')

# Botão para realizar a consulta
if st.button('Consultar'):
    if query.strip():
        question = prefix_query + query
        answer = chain.invoke(question)
        print('Resposta: ', answer)
        st.write(answer['result'])
    else:
        st.warning("Por favor, digite uma pergunta antes de consultar.")