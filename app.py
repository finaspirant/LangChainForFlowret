import streamlit as st
from streamlit_chat import message

from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA

#Load API Keys
envLoad = load_dotenv('.env')
print(envLoad)

#load Embeddings
embeddings = OpenAIEmbeddings()


db = DeepLake(dataset_path="hub://shailfinaspirant/flowret-algorithm", read_only=True, embedding_function=embeddings)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10



model = ChatOpenAI(model='gpt-3.5-turbo') # switch to 'gpt-4' with money
qa = RetrievalQA.from_llm(model, retriever=retriever)

# Return the result of the query
qa.run("What is the repository's name?")


st.title(f"Chat with GitHub Repository")
# Initialize the session state for placeholder messages.
if "generated" not in st.session_state:
	st.session_state["generated"] = ["i am ready to help you ser"]

if "past" not in st.session_state:
	st.session_state["past"] = ["hello"]

# A field input to receive user queries
user_input = st.text_input("", key="input")

# Search the database and add the responses to state
if user_input:
	output = qa.run(user_input)
	st.session_state.past.append(user_input)
	st.session_state.generated.append(output)

# Create the conversational UI using the previous states
if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
        message(st.session_state["generated"][i], key=str(i))




