from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

#Load API Keys
envLoad = load_dotenv('.env')
print(envLoad)

#load Embeddings
embeddings = OpenAIEmbeddings()
#print(embeddings)

db = DeepLake(dataset_path="hub://shailfinaspirant/flowret-algorithm", read_only=True, embedding_function=embeddings)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10



model = ChatOpenAI(model='gpt-3.5-turbo') # switch to 'gpt-4'
qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)

questions = [
    "what does constructor of Step.java do?",
    "what does the method changeWorkBasket() does?",
    "What does Floweret code do?",
    "what are the models in Flowret?",
    "What does EventHandler.java does?",
]
chat_history = []

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")