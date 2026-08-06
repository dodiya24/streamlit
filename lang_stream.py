import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import streamlit as st

load_dotenv()
api_key = os.getenv("Groq_API_Key")

st.set_page_config(
    page_title="AI Chatbot", 
    page_icon="🤖", 
    layout="wide"
)
st.title("🤖 AI Assistant with Live Evaluation")
st.write("Ask a question, get an AI answer, and view the automated quality evaluation score!")

# LLMs Setup
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.7, 
    max_retries=3,
    groq_api_key=api_key
)
eval_llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0.1, 
    max_retries=3,
    groq_api_key=api_key
)

# Main Chat Chain
SYSTEM_PROMPT = "You are an expert AI assistant. Be concise, technical, and helpful."
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# Evaluator Chain
eval_chain = ChatPromptTemplate.from_messages([
    ("system", "You are AI assistant, Return ONLY valid JSON. Do not include markdown codeblocks like ```json."),
    ("human", 'Question: {question}\nAnswer: {answer}\n'
              'Score 0.0 to 1.0 on: relevance, coherence, conciseness.\n'
              'Return strictly in this format: '
              '{{"relevance": 0.0, "coherence": 0.0, "conciseness": 0.0, "feedback": "..."}}')
]) 

eval_chain_llm = eval_chain | eval_llm | JsonOutputParser()

# Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # For LangChain context

if "messages" not in st.session_state:
    st.session_state.messages = []       # For Streamlit UI display

# Render Previous Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "eval" in msg and msg["eval"]:
            e = msg["eval"]
            st.info(f"📊 **Overall Score:** {e['overall']} / 1.0  \n"
                    f"**Feedback:** {e['feedback']}  \n"
                    f"*(Relevance: {e['relevance']} | Coherence: {e['coherence']} | Conciseness: {e['conciseness']})*")

# User Input Processing
if user_input := st.chat_input("Ask something..."):

    # Display User Input
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate Main Response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Stream response word-by-word
        for chunk in chain.stream({"input": user_input, "history": st.session_state.chat_history}):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)

        # Pass User Prompt & AI Answer to Eval LLM
        eval_result = None
        with st.spinner("Evaluating response quality..."):
            try:
                # 💡 FIXED: eval_chain ki jagah eval_chain_llm use kiya hai
                scores = eval_chain_llm.invoke({"question": user_input, "answer": full_response})
                
                rel = scores.get("relevance", 0)
                coh = scores.get("coherence", 0)
                con = scores.get("conciseness", 0)
                overall = round((rel + coh + con) / 3, 2)

                eval_result = {
                    "relevance": rel,
                    "coherence": coh,
                    "conciseness": con,
                    "overall": overall,
                    "feedback": scores.get("feedback", "N/A")
                }

                # Display Evaluation Box
                st.info(f"📊 **Overall Score:** {overall} / 1.0  \n"
                        f"**Feedback:** {eval_result['feedback']}  \n"
                        f"*(Relevance: {rel} | Coherence: {coh} | Conciseness: {con})*")

            except Exception as err:
                st.warning(f"Evaluation could not be generated. Error: {err}")

    # Save to Memory for multi-turn chat context
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=full_response))

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "eval": eval_result
    })