import os
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

llm = OpenAI(temperature=0, streaming=True, openai_api_key=os.getenv('OPENAI_API_KEY'))
tools = load_tools(["ddg-search"])
agent = initialize_agent(
    tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st.write("🧠 thinking...")
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_callback])
        st.write(response)