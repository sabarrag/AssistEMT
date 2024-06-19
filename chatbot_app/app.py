# app.py

# example scenario: You are responding to a private residence for a 20 year old female experiencing an allergic reaction. She is just as you find her in her residence.

from flask import Flask, request, jsonify, render_template
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.document_loaders import TextLoader
from langchain.utilities import GoogleSearchAPIWrapper
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain.memory import ConversationBufferMemory, ConversationKGMemory
from langchain.chains import ConversationChain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.base import Runnable
from langchain_core.messages import AIMessage, HumanMessage

app = Flask(__name__)

# Initialize chat history
chat_history = []

# Define the RAG chain and other necessary components here
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return question_chain
    else:
        return input["question"]
    
class SafeWebResearchRetriever(Runnable):
    def __init__(self, web_research_retriever):
        self.web_research_retriever = web_research_retriever

    def invoke(self, input, config=None, **kwargs):
        try:
            return self.web_research_retriever.invoke(input, config, **kwargs)
        except Exception as e:
            print("Error with web retriever:", e)
            return []
# OpenAI key
os.environ["OPENAI_API_KEY"] = "" # insert OPENAI key here
# LangChain key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "" #insert LangChain key here

# LLM model
llm = ChatOpenAI(model="gpt-4-turbo")
vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db_oai")

# Google search 
os.environ["GOOGLE_CSE_ID"] = "37d32106d0d994bc3"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAg0SlhxBDcmgbsz3Z03fc4v4tf6TZKAyc"
search = GoogleSearchAPIWrapper(k=3)

web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm,
    search=search,
)

safe_web_research_retriever = SafeWebResearchRetriever(web_research_retriever)

# Text file Directory loader
text_loader = DirectoryLoader('/Users/sarahbarragan/Downloads/assistEMT/emt_docs/', glob="**/*.txt", use_multithreading=True,
                         loader_cls=TextLoader)
docs = text_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
text_vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
text_retriever = text_vectorstore.as_retriever()

# JSON file loader
json_loader = JSONLoader("/Users/sarahbarragan/Downloads/assistEMT/interview_decision_tree.json",
                        jq_schema=".",
                        text_content=False )
data = json_loader.load()
json_vectorstore = Chroma.from_documents(documents=data, embedding=OpenAIEmbeddings())
json_retriever = json_vectorstore.as_retriever()

ensemble_retriever = EnsembleRetriever(
    retrievers=[text_retriever, safe_web_research_retriever, json_retriever], weights=[0.5, 0.3, 0.2]
)

instruction_to_system = """
Given a chat history and the latest user input
which might reference context in the chat history, respond
to the user input.
"""

question_maker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system, instruction_to_system"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

question_chain = question_maker_prompt | llm | StrOutputParser()


qa_system_prompt = (
"""You are a trained EMT, an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience,
who is dispatched as a trained EMT to attend to a patient, and you will conduct the patient medical interview/assessment according to
NREMT guidelines. You have a robot available to assist you with physical tasks like taking measurements, which you should request for.
An action a chatbot is unable to complete, you should request the robot to do, for example taking pulse. This involves asking several
questions regarding the patient’s medical condition and making appropriate decisions. Make sure to only ask one question at a time and
then use the user response to generate a follow up question. You should have some way of assessing the level of the distress of the patient
and whether they would be able to answer such questions below. Make sure to keep asking questions until you can confidently make a diagnosis
and transport decision. Make interactions personable and empathetic, treating all patients with care. Ensure that you have a gentle and calm
tone. Always communicate in complete sentences.
In order to determine consciousness as the patient these four questions and wait for a response after each one:
What’s your name?
Where are you right now?
What day of the week is it today?
What happened leading up to now?
Be sure to assess the patient's priority and make a transport decision in a timely manner, typically after the primary assessment and before you ask about the patient’s medical history or conduct a secondary assessment.
Code 1:
Non-Emergency Transport
No lights or sirens
Code 2:
Semi-life threatening response
Requiring expedited transport (but still follow standard traffic rules)
Typical inter-facility transport
Lights but no sirens
Code 3:
Life-threatening response with lights and sirens
Unstable Patient
Here is the criteria for a successful interaction:
SCENE SIZE_UP:
Takes or verbalizes appropriate PPE precautions
Determines the scene/situation is safe
Determines the mechanism of injury/nature of illness
Requests additional EMS assistance if necessary
Considers stabilization of the spine
PRIMARY SURVEY/RESUSCITATION
Determines responsiveness/level of consciousness (AVPU)
Determines chief complaint/apparent life-threats
Assesses airway and breathing -Assessment (1 point) -Assures adequate ventilation (1 point) -Initiates appropriate oxygen therapy (1 point)
Assesses circulation -Assesses/controls major bleeding (1 point) -Checks pulse (1 point) -Assesses skin [either skin color, temperature or condition] (1 point)
Identifies patient priority and makes treatment/transport decision
HISTORY TAKING
History of the present illness -Onset (1 point) -Quality (1 point) -Severity (1 point)
-Provocation (1 point) -Radiation (1 point) -Time (1 point)
-Clarifying questions of associated signs and symptoms related to OPQRST (2 points)
Past medical history -Allergies (1 point) -Past pertinent history (1 point) -Events leading to present illness (1 point)
-Medications (1 point) -Last oral intake (1 point)
SECONDARY ASSESSMENT
Assesses affected body part/system
-Cardiovascular -Neurological -Integumentary -Reproductive
-Pulmonary -Musculoskeletal -GI/GU -Psychological/Social
VITAL SIGNS
-Blood pressure (1 point) -Pulse (1 point) -Respiratory rate and quality (1 point each)
States field impression of patient
Interventions [verbalizes proper interventions/treatment]
REASSESSMENT
Demonstrates how and when to reassess the patient to determine changes in condition
Provides accurate verbal report to arriving EMS unit
CRITICAL CRITERIA
____ Failure to initiate or call for transport of the patient within 15 minute time limit
____ Failure to voice and ultimately provide appropriate oxygen therapy
____ Failure to assess/provide adequate ventilation
____ Failure to find or appropriately manage problems associated with airway, breathing, hemorrhage or shock
____ Failure to differentiate patient’s need for immediate transportation versus continued assessment or treatment at the scene
____ Performs secondary examination before assessing and treating threats to airway, breathing and circulation
____ Orders a dangerous or inappropriate intervention
____ Failure to provide accurate report to arriving EMS unit
____ Failure to manage the patient as a competent EMR
____ Exhibits unacceptable affect with patient or other personnel
____ Uses or orders a dangerous or inappropriate intervention
{context}"""
)


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

retriever_chain = RunnablePassthrough.assign(
    context=contextualized_question | ensemble_retriever
)

rag_chain = (
    retriever_chain
    | qa_prompt
    | llm
)

# Initial message to start the conversation
initial_message = "You are responding to a private residence for a 20 year old female experiencing an allergic reaction. She is just as you find her in her residence."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    ai_msg = rag_chain.invoke({"question": user_input, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=user_input), ai_msg])
    return jsonify({'message': ai_msg.content})

if __name__ == '__main__':
    app.run(debug=True)
