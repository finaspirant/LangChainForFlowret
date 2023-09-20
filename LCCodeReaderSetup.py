from dotenv import load_dotenv
import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

#Load API Keys
envLoad = load_dotenv('.env')
print(envLoad)

#load Embeddings
embeddings = OpenAIEmbeddings()
#print(embeddings)


root_dir = 'flowret'
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

username = "shailfinaspirant" # replace with your username from app.activeloop.ai
db = DeepLake(dataset_path=f"hub://{username}/flowret-algorithm",  embedding_function=embeddings)
db.add_documents(texts)
