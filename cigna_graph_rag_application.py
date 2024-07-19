import os
import streamlit as st
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from custom_scripts.cypher import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import messages_to_dict
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Define prompt templates - Cypher generation prompt and Question answering prompt
CYPHER_GENERATION_TEMPLATE = """
**Task** 
Generate valid Cypher statement to query a graph database.

**Description of 'Plan' node properties**

#id: Unique identifier and name of the medical plan
#monthly_premium: Monthly cost of the medical plan
#deductible: Deductible amount for the medical plan
#max_oop: Out-of-pocket maximum of the medical plan
#estimated: Estimated annual cost of the medical plan
#network: In-network of Health Maintenance Organization (HMO)
#primary_care: Primary care visit copay/coinsurance
#specialist: Specialist visit copay/coinsurance
#generic_dr: Cost of a generic drug prescription
#laboratory: Cost of laboratory services
#emergency: Emergency room visit copay/coinsurance
#urgent_care: Urgent care visit copay/coinsurance
#hospital_stay: Cost per night for a hospital stay

**Instructions**
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{{schema}}

**Cypher examples**

#question: `Which medical plans have a deductible less than 1000?`
#cypher query: MATCH (p:Plan) WHERE p.deductible < 1000 RETURN p.id

#question: `What are the top 3 plans with the lowest monthly premiums?`
#cypher query: `MATCH (p:Plan) RETURN p.id, p.monthly_premium ORDER BY p.monthly_premium ASC LIMIT 3`

#question: `Which medical plan has the lowest deductible?`
#cypher query: `MATCH (p:Plan) RETURN p.id ORDER BY p.deductible ASC LIMIT 1`

#question: `Which medical plan has the lowest annual estimated cost?`
#cypher query: `MATCH (p:Plan) RETURN p.id ORDER BY p.estimated ASC LIMIT 1`

#question: `What is the plan with the lowest copay for an urgent care visit?`
#cypher query: `MATCH (p:Plan) RETURN p.id, p.urgent_care ORDER BY p.urgent_care ASC LIMIT 1`

#question(out of scope scenario): `Who is the president of USA`
#cypher query: `MATCH (n) WHERE 1 = 0 RETURN n`

The question is: {{question}}

Below is the chat history between User and Assistant. Please use chat history to extract required information,

#chat history:

`{chat_history}`
"""

QA_TEMPLATE = """
Your role is to provide understandable and pleasant information about Cigna insurance plans. Use the details provided to construct your response. 
However, if no information is provided and if the user's question is not insurance related, remind them by saying, "I am dedicated to providing information about Cigna medical plans to help you choose the most suitable insurance plan for yourself."
If user's question is greeting, please answer in convincing manner.
Final answer should be easily readable and structured.

#Information:
{{context}}

#Question: {{question}}

Please note that if there any context available, you must generate the answer.

Following are the examples of question, context and answer. Please use following examples to generate your answer.
#Question: What are the plans with a monthly premium less than $300?
#Information: []
#Answer: `There is no plan with a monthly premium less than $300`

Below is the chat history between User and Assistant. Please use chat history to extract required information,

#chat history:
`{chat_history}`

Helpful Answer:"""

# Define prompts
# CYPHER_GENERATION_PROMPT = PromptTemplate(template=CYPHER_GENERATION_TEMPLATE)
# QA_PROMPT = PromptTemplate(template=QA_TEMPLATE)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", input_key="query", output_key="result", chat_memory=msgs, return_messages=True)

def generate_chat_history_from_memory(memory_messages):
    """
    Sample object: [AIMessage(content='Hello! Welcome to Cigna Health Plan Assistant. What can I help you with today?'), 
    HumanMessage(content='what is the best medical plan?'), 
    AIMessage(content='Based on the provided information, the available medical plan is')]
    """
    chat_history_str = ""
    for msg_obj in memory_messages:
        msg_type = msg_obj['type']
        if msg_type == 'ai':
            msg_type = 'Assistant'
        else:
            msg_type = 'User'
        msg_text = msg_obj['data']['content']
        chat_history_str = chat_history_str + msg_type + ": "+ msg_text + "\n"
    return chat_history_str


def get_answer_from_graph(query, memory, cypher_gp, qa_temp):
    """
    This function is for generate the answer for the customer's query using graph qa chain
    """
    # Generate the answer for customer query
    chat_history_str = generate_chat_history_from_memory(messages_to_dict(memory.chat_memory.messages))
    # Add conversation memory to prompts
    cypher_gp = cypher_gp.format(chat_history=chat_history_str)
    qa_temp = qa_temp.format(chat_history=chat_history_str)
    # Load prompt templates
    CYPHER_GENERATION_PROMPT = PromptTemplate(template=cypher_gp)
    QA_PROMPT = PromptTemplate(template=qa_temp)
    try:
        # Initiate the QA chain
        chain_with_examples = GraphCypherQAChain.from_llm(
            llm=llm, graph=graph, verbose=True,
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            # cypher_llm_kwargs={"prompt":CYPHER_GENERATION_PROMPT},
            qa_prompt=QA_PROMPT,
            memory=memory,
            return_intermediate_steps=True
        )
        chain_response = chain_with_examples.invoke({"query": query})
    except Exception as e:
        print(e)
    return chain_response

# Streamlit Configuration
st.set_page_config(page_title="Cigna Health Plan Assistant")
st.title("Cigna Health Plan Assistant")

if len(msgs.messages) == 0:
    msgs.clear()
    msgs.add_ai_message("Hello! Welcome to Cigna Health Plan Assistant. What can I help you with today?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            print('Steps :',step)
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)


if prompt := st.chat_input(placeholder="Enter health plans related question here"):
    st.chat_message("user").write(prompt)
    # Initialize LLM and graph
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    graph = Neo4jGraph()

    with st.chat_message("assistant"):
        response = get_answer_from_graph(prompt, memory, CYPHER_GENERATION_TEMPLATE, QA_TEMPLATE)
        # print("Response : ",response)
        st.markdown(response['result'])
        # print(msgs.messages)
        # st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]
