# # main.py

# # ==============================================================================
# # ⚠️ CRITICAL FIX FOR ASYNCIO ON WINDOWS V2
# # This block MUST be at the very top of the script.
# # ==============================================================================
# import sys
# import asyncio

# if sys.platform == "win32":
#     try:
#         loop = asyncio.get_running_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
# # ==============================================================================

# import streamlit as st
# import os
# import pypdf
# import faiss  # Required for ParentDocumentRetriever with FAISS

# # LangChain & community imports
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # Switched from Google to OpenAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.storage import InMemoryStore  # For parent docstore
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain import hub
# from langchain.agents import create_react_agent, AgentExecutor
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.retrievers import ParentDocumentRetriever
# from langchain.docstore.document import Document
# from langchain_community.docstore.in_memory import InMemoryDocstore  # Updated import

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Professional LangChain Agent (OpenAI)",
#     page_icon="🤖",
#     layout="wide",
# )

# # --- API Keys ---
# try:
#     openai_api_key = os.environ["OPENAI_API_KEY"]
#     tavily_api_key = os.environ["TAVILY_API_KEY"]
# except KeyError:
#     st.error("🔴 **Error:** Make sure both OPENAI_API_KEY and TAVILY_API_KEY are set as environment variables.")
#     st.stop()

# # --- UI & Styling ---
# st.title("🤖 Professional Conversational Agent (OpenAI)")
# st.markdown("""
# This agent can have a conversation with you about a document you upload.
# - **Upload a PDF** in the sidebar to create a knowledge base.
# - It uses an advanced **Parent Document Retriever** for more accurate answers.
# - It **remembers** your conversation for follow-up questions.
# - It **shows its work** so you can see its reasoning process.
# """)

# # --- Agent Setup ---
# # Initialize LLM
# llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0)

# # Session State Initialization
# if "store" not in st.session_state:
#     st.session_state.store = {}  # Stores chat histories
# if "agent_with_chat_history" not in st.session_state:
#     st.session_state.agent_with_chat_history = None
# if "retriever" not in st.session_state:
#     st.session_state.retriever = None

# # Function to get or create a chat history for a session
# def get_session_history(session_id: str):
#     if session_id not in st.session_state.store:
#         st.session_state.store[session_id] = ChatMessageHistory()
#     return st.session_state.store[session_id]

# # --- Sidebar for PDF Upload and Agent Creation ---
# with st.sidebar:
#     st.header("🧠 Knowledge Base")
#     st.markdown("Note: You may need to install `faiss-cpu` (`pip install faiss-cpu`) for this to work.")
#     uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

#     if st.button("Create Agent"):
#         if uploaded_file is not None:
#             with st.spinner("Processing PDF and creating agent... This may take a moment."):
#                 try:
#                     # 1. Load and Process the PDF
#                     pdf_reader = pypdf.PdfReader(uploaded_file)
#                     raw_text = ""
#                     for page in pdf_reader.pages:
#                         raw_text += page.extract_text() or ""

#                     docs = [Document(page_content=raw_text)]

#                     # 2. Setup the Parent Document Retriever
#                     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#                     embedding_dimension = 1536  # e.g., for text-embedding-3-small

#                     # Create FAISS index
#                     index = faiss.IndexFlatL2(embedding_dimension)

#                     parent_docstore = InMemoryStore()
#                     child_docstore = InMemoryDocstore()

#                     vectorstore = FAISS(
#                         embedding_function=embeddings,
#                         index=index,
#                         docstore=child_docstore,
#                         index_to_docstore_id={}
#                     )

#                     parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
#                     child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)

#                     st.session_state.retriever = ParentDocumentRetriever(
#                         vectorstore=vectorstore,
#                         docstore=parent_docstore,
#                         child_splitter=child_splitter,
#                         parent_splitter=parent_splitter,
#                     )
#                     st.session_state.retriever.add_documents(docs, ids=None)
#                     st.success("PDF processed and retriever created!")

#                     # 3. Create Tools
#                     retriever_tool = create_retriever_tool(
#                         st.session_state.retriever,
#                         "pdf_document_qa",
#                         "MUST USE this tool for any questions about the content of the uploaded PDF document."
#                     )
#                     search_tool = TavilySearchResults(api_key=tavily_api_key)
#                     tools = [retriever_tool, search_tool]

#                     # 4. Create the Agent
#                     prompt = hub.pull("hwchase17/react-chat")
#                     agent = create_react_agent(llm, tools, prompt)
#                     agent_executor = AgentExecutor(
#                         agent=agent,
#                         tools=tools,
#                         verbose=True,
#                         handle_parsing_errors=True,
#                         return_intermediate_steps=True
#                     )

#                     # 5. Attach Memory
#                     st.session_state.agent_with_chat_history = RunnableWithMessageHistory(
#                         agent_executor,
#                         get_session_history,
#                         input_messages_key="input",
#                         history_messages_key="chat_history",
#                     )
#                     st.success("Conversational agent is ready!")

#                 except Exception as e:
#                     st.error(f"An error occurred: {e}")
#         else:
#             st.warning("Please upload a PDF file first.")

# # --- Main Chat Interface ---
# st.header("💬 Chat with the Professional Agent")

# # Display existing chat history
# chat_history = get_session_history("main_chat")
# for msg in chat_history.messages:
#     st.chat_message(msg.type).write(msg.content)

# # Handle new user input
# if user_question := st.chat_input("Ask a question..."):
#     st.chat_message("user").write(user_question)

#     if st.session_state.agent_with_chat_history is None:
#         st.chat_message("assistant").warning("Please create the agent using the sidebar first.")
#     else:
#         with st.chat_message("assistant"):
#             with st.spinner("Agent is thinking..."):
#                 try:
#                     result = st.session_state.agent_with_chat_history.invoke(
#                         {"input": user_question},
#                         config={"configurable": {"session_id": "main_chat"}}
#                     )
#                     st.markdown(result["output"])
#                     with st.expander("Show agent's thought process"):
#                         st.json(result["intermediate_steps"])
#                 except Exception as e:
#                     st.error(f"An error occurred: {e}")

# # --- Footer ---
# st.sidebar.markdown("---")
# st.sidebar.markdown("Built with ❤️ using [Streamlit](https://streamlit.io) and [LangChain](https://www.langchain.com/).")







# # main.py

# # ==============================================================================
# # CRITICAL FIX FOR ASYNCIO ON WINDOWS V2
# # This block MUST be at the very top of the script.
# # ==============================================================================
# import sys
# import asyncio

# if sys.platform == "win32":
#     try:
#         loop = asyncio.get_running_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)
# # ==============================================================================

# import streamlit as st
# import os
# import pypdf
# import faiss # Required for ParentDocumentRetriever with FAISS
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.storage import InMemoryStore
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain import hub
# from langchain.agents import create_react_agent, AgentExecutor
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.retrievers import ParentDocumentRetriever
# from langchain.docstore.document import Document
# import uuid

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Professional LangChain Agent (OpenAI)",
#     page_icon="🤖",
#     layout="wide",
# )

# # --- API Keys ---
# try:
#     openai_api_key = os.environ["OPENAI_API_KEY"]
#     tavily_api_key = os.environ["TAVILY_API_KEY"]
# except KeyError:
#     st.error("🔴 **Error:** Make sure both OPENAI_API_KEY and TAVILY_API_KEY are set as environment variables.")
#     st.stop()

# # --- UI & Styling ---
# st.title("🤖 Professional Conversational Agent (OpenAI)")
# st.markdown("""
# This agent can have a conversation with you about a document you upload.
# - **Upload a PDF** in the sidebar to create a knowledge base.
# - It uses an advanced **Parent Document Retriever** for more accurate answers.
# - It **rememebers** your conversation for follow-up questions.
# - It **shows its work** so you can see its reasoning process.
# """)

# # --- Agent Setup ---

# # Initialize LLM
# llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0)

# # Session State Initialization
# if "store" not in st.session_state:
#     st.session_state.store = {} # Stores chat histories
# if "agent_with_chat_history" not in st.session_state:
#     st.session_state.agent_with_chat_history = None
# if "retriever" not in st.session_state:
#     st.session_state.retriever = None

# # Function to get or create a chat history for a session
# def get_session_history(session_id: str):
#     if session_id not in st.session_state.store:
#         st.session_state.store[session_id] = ChatMessageHistory()
#     return st.session_state.store[session_id]

# # --- Sidebar for PDF Upload and Agent Creation ---
# with st.sidebar:
#     st.header("🧠 Knowledge Base")
#     st.markdown("Note: You may need to install `faiss-cpu` (`pip install faiss-cpu`) for this to work.")
#     uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

#     if st.button("Create Agent"):
#         if uploaded_file is not None:
#             with st.spinner("Processing PDF and creating agent... This may take a moment."):
#                 try:
#                     # 1. Load and Process the PDF
#                     pdf_reader = pypdf.PdfReader(uploaded_file)
#                     raw_text = ""
#                     for page in pdf_reader.pages:
#                         raw_text += page.extract_text()
                    
#                     docs = [Document(page_content=raw_text)]
                    
#                     # 2. Setup the Parent Document Retriever - MORE ROBUST METHOD
#                     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                    
#                     parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
#                     child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)

#                     # Split the documents into parent and child chunks
#                     parent_docs = parent_splitter.split_documents(docs)
                    
#                     child_docs = []
#                     parent_doc_ids = []
                    
#                     for i, doc in enumerate(parent_docs):
#                         _id = str(uuid.uuid4())
#                         child_doc_chunks = child_splitter.split_documents([doc])
#                         for chunk in child_doc_chunks:
#                             chunk.metadata["parent_id"] = _id
#                         child_docs.extend(child_doc_chunks)
#                         parent_doc_ids.append((_id, doc))

#                     # Create the vectorstore from the child documents
#                     vectorstore = FAISS.from_documents(child_docs, embeddings)
                    
#                     # Create the parent document store and add the parent documents
#                     docstore = InMemoryStore()
#                     docstore.mset(parent_doc_ids)
                    
#                     # Initialize the retriever
#                     st.session_state.retriever = ParentDocumentRetriever(
#                         vectorstore=vectorstore,
#                         docstore=docstore,
#                         child_splitter=child_splitter, # Not used in this setup, but good practice to pass
#                         parent_splitter=parent_splitter, # Not used in this setup, but good practice to pass
#                     )
                    
#                     st.success("PDF processed and retriever created!")

#                     # 3. Create the Tools
#                     retriever_tool = create_retriever_tool(
#                         st.session_state.retriever,
#                         "pdf_document_qa",
#                         "MUST USE this tool for any questions about the content of the uploaded PDF document. This tool has exclusive information.",
#                     )
#                     search_tool = TavilySearchResults(api_key=tavily_api_key)
#                     tools = [retriever_tool, search_tool]

#                     # 4. Create the Agent
#                     prompt = hub.pull("hwchase17/react-chat")
#                     agent = create_react_agent(llm, tools, prompt)
#                     agent_executor = AgentExecutor(
#                         agent=agent, 
#                         tools=tools, 
#                         verbose=True, 
#                         handle_parsing_errors=True,
#                         return_intermediate_steps=True
#                     )

#                     # 5. Add Memory
#                     st.session_state.agent_with_chat_history = RunnableWithMessageHistory(
#                         agent_executor,
#                         get_session_history,
#                         input_messages_key="input",
#                         history_messages_key="chat_history",
#                     )
                    
#                     st.success("Conversational agent is ready!")
#                 except Exception as e:
#                     st.error(f"An error occurred: {e}")
#         else:
#             st.warning("Please upload a PDF file first.")

# # --- Main Chat Interface ---
# st.header("💬 Chat with the Professional Agent")

# # Display chat history
# chat_history = get_session_history("main_chat")
# for msg in chat_history.messages:
#     st.chat_message(msg.type).write(msg.content)

# # Chat input
# if user_question := st.chat_input("Ask a question..."):
#     with st.chat_message("user"):
#         st.markdown(user_question)

#     if st.session_state.agent_with_chat_history is None:
#         with st.chat_message("assistant"):
#             st.warning("Please create the agent using the sidebar first.")
#     else:
#         with st.chat_message("assistant"):
#             with st.spinner("Agent is thinking..."):
#                 try:
#                     # Invoke the agent with memory and get intermediate steps
#                     response = st.session_state.agent_with_chat_history.invoke(
#                         {"input": user_question},
#                         config={"configurable": {"session_id": "main_chat"}},
#                     )
                    
#                     st.markdown(response["output"])

#                     # Display the agent's thought process in an expander
#                     with st.expander("Show agent's thought process"):
#                         st.json(response["intermediate_steps"])

#                 except Exception as e:
#                     st.error(f"An error occurred: {e}")

# # --- Footer ---
# st.sidebar.markdown("---")
# st.sidebar.markdown("Built with ❤️ using [Streamlit](https://streamlit.io) and [LangChain](https://www.langchain.com/).")







# main.py
# ==============================================================================
# FINAL DEPLOYABLE PRODUCT
#
# To run this application:
# 1. Install all required libraries:
#    pip install streamlit pypdf faiss-cpu langchain-openai langchain tavily-python langchainhub
#
# 2. Create a `requirements.txt` file with the following content:
#    streamlit
#    pypdf
#    faiss-cpu
#    langchain-openai
#    langchain
#    tavily-python
#    langchainhub
#
# 3. Set your API keys as environment variables. You can do this in your
#    terminal or by creating a `.env` file and using a library like `python-dotenv`.
#    - OPENAI_API_KEY="sk-..."
#    - TAVILY_API_KEY="tvly-..."
#
# ==============================================================================





# # --- Core Imports ---
# import streamlit as st
# import os
# import pypdf
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain import hub
# from langchain.agents import create_react_agent, AgentExecutor
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.docstore.document import Document

# # --- Asyncio Fix for Streamlit and LangChain on Windows ---
# # This is a critical fix for running LangChain's async components in Streamlit on Windows.
# import sys
# import asyncio
# if sys.platform == "win32":
#     try:
#         loop = asyncio.get_running_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Deployable AI Agent",
#     page_icon="🚀",
#     layout="wide",
# )

# # --- API Key Setup ---
# try:
#     openai_api_key = os.environ["OPENAI_API_KEY"]
#     tavily_api_key = os.environ["TAVILY_API_KEY"]
# except KeyError:
#     st.error("🔴 **Error:** API keys not found. Please set OPENAI_API_KEY and TAVILY_API_KEY as environment variables.")
#     st.stop()

# # --- UI & Styling ---
# st.title("🚀 Deployable Conversational AI Agent")
# st.markdown("""
# Welcome to your professional-grade AI assistant! This agent leverages advanced retrieval techniques and conversational memory to provide accurate, context-aware answers.

# **Key Features:**
# - **📚 PDF-Powered Knowledge Base:** Upload a PDF to give the agent specialized knowledge.
# - **🧠 Conversational Memory:** Ask follow-up questions and the agent will remember the context.
# - **🔍 Transparent Reasoning:** See the agent's step-by-step thought process for every answer.
# - **⚡ Real-time Streaming:** Responses are streamed token-by-token for a dynamic experience.
# """)

# # --- Agent and Tool Setup ---

# # Initialize the LLM with streaming enabled
# llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0, streaming=True)

# # Session State Initialization
# # This ensures that data persists across user interactions in the Streamlit app.
# if "store" not in st.session_state:
#     st.session_state.store = {} # Stores chat histories for different sessions
# if "agent_with_chat_history" not in st.session_state:
#     st.session_state.agent_with_chat_history = None
# if "retriever" not in st.session_state:
#     st.session_state.retriever = None

# # Function to get or create a chat history for a session
# def get_session_history(session_id: str):
#     if session_id not in st.session_state.store:
#         st.session_state.store[session_id] = ChatMessageHistory()
#     return st.session_state.store[session_id]

# # --- Sidebar for PDF Upload and Agent Creation ---
# with st.sidebar:
#     st.header("⚙️ Configuration")
#     st.markdown("Upload a PDF document to create a specialized knowledge base for the agent.")
    
#     uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

#     if st.button("Create Agent"):
#         if uploaded_file is not None:
#             with st.spinner("Processing PDF and creating agent... This is a one-time setup per document."):
#                 try:
#                     # 1. Load and Process the PDF
#                     pdf_reader = pypdf.PdfReader(uploaded_file)
#                     raw_text = "".join(page.extract_text() for page in pdf_reader.pages)
#                     docs = [Document(page_content=raw_text)]

#                     # 2. Refined Retriever Setup (Standard, well-tuned retriever)
#                     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#                     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
#                     splits = text_splitter.split_documents(docs)
#                     vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
#                     st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                    
#                     st.success("PDF processed and knowledge base created!")

#                     # 3. Create the Tools
#                     retriever_tool = create_retriever_tool(
#                         st.session_state.retriever,
#                         "pdf_knowledge_base",
#                         "MUST USE this tool for any questions about the content of the uploaded PDF document. This tool contains exclusive information.",
#                     )
#                     search_tool = TavilySearchResults(api_key=tavily_api_key)
#                     tools = [retriever_tool, search_tool]

#                     # 4. Create the Agent
#                     prompt = hub.pull("hwchase17/react-chat")
#                     agent = create_react_agent(llm, tools, prompt)
#                     agent_executor = AgentExecutor(
#                         agent=agent, 
#                         tools=tools, 
#                         verbose=True, 
#                         handle_parsing_errors=True,
#                         return_intermediate_steps=True # Crucial for showing the agent's work
#                     )

#                     # 5. Add Memory
#                     st.session_state.agent_with_chat_history = RunnableWithMessageHistory(
#                         agent_executor,
#                         get_session_history,
#                         input_messages_key="input",
#                         history_messages_key="chat_history",
#                     )
                    
#                     st.success("Conversational agent is ready to chat!")
#                 except Exception as e:
#                     st.error(f"An error occurred during agent creation: {e}")
#         else:
#             st.warning("Please upload a PDF file first.")

# # --- Main Chat Interface ---

# # Async function to handle the streaming response and display intermediate steps
# async def process_user_question(user_question):
#     intermediate_steps = []
#     full_response = ""

#     # Use a Streamlit container for the final answer, so we can update it
#     answer_container = st.empty()
    
#     # Use a Streamlit expander for the thought process
#     with st.expander("Show agent's thought process..."):
#         steps_container = st.empty()

#     async for event in st.session_state.agent_with_chat_history.astream_events(
#         {"input": user_question},
#         config={"configurable": {"session_id": "main_chat"}},
#         version="v2"
#     ):
#         kind = event["event"]
        
#         if kind == "on_chain_start":
#             steps_container.markdown(f"**Thinking...**")

#         elif kind == "on_tool_start":
#             intermediate_steps.append(f"**Tool Used:** `{event['name']}` with input: `{event['data'].get('input')}`")
#             steps_container.markdown("\n\n".join(intermediate_steps))

#         elif kind == "on_tool_end":
#             output = event['data'].get('output')
#             if output:
#                 # Truncate long tool outputs for cleaner display
#                 display_output = (output[:300] + '...') if len(output) > 300 else output
#                 intermediate_steps.append(f"**Tool Output:**\n\n```\n{display_output}\n```")
#                 steps_container.markdown("\n\n".join(intermediate_steps))
        
#         elif kind == "on_chat_model_stream":
#             content = event["data"]["chunk"].content
#             if content:
#                 # --- FIX: Append content to a string variable ---
#                 full_response += content
#                 # Update the container with the full response and a blinking cursor
#                 answer_container.markdown(full_response + "▌")

#     # Once streaming is done, remove the cursor
#     answer_container.markdown(full_response)

# st.header("💬 Chat Interface")

# # Display previous chat messages
# chat_history = get_session_history("main_chat")
# for msg in chat_history.messages:
#     st.chat_message(msg.type).write(msg.content)

# # Get new user input
# if user_question := st.chat_input("Ask a question about your document or a general topic..."):
#     with st.chat_message("user"):
#         st.markdown(user_question)

#     if st.session_state.agent_with_chat_history is None:
#         with st.chat_message("assistant"):
#             st.warning("Please create the agent using the sidebar first.")
#     else:
#         with st.chat_message("assistant"):
#             # Run the async function to process and display the response
#             asyncio.run(process_user_question(user_question))




# main.py
# ==============================================================================
# FINAL DEPLOYABLE PRODUCT
#
# To run this application:
# 1. Install all required libraries:
#    pip install streamlit pypdf faiss-cpu langchain-openai langchain tavily-python langchainhub
#
# 2. Create a `requirements.txt` file with the following content:
#    streamlit
#    pypdf
#    faiss-cpu
#    langchain-openai
#    langchain
#    tavily-python
#    langchainhub
#
# 3. Set your API keys as environment variables.
#    - OPENAI_API_KEY="sk-..."
#    - TAVILY_API_KEY="tvly-..."
#
# ==============================================================================








# # --- Core Imports ---
# import streamlit as st
# import os
# import pypdf
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain import hub
# from langchain.agents import create_react_agent, AgentExecutor
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.docstore.document import Document

# # --- Asyncio Fix for Streamlit and LangChain on Windows ---
# import sys
# import asyncio
# if sys.platform == "win32":
#     try:
#         loop = asyncio.get_running_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Deployable AI Agent",
#     page_icon="🚀",
#     layout="wide",
# )

# # --- API Key Setup ---
# try:
#     openai_api_key = os.environ["OPENAI_API_KEY"]
#     tavily_api_key = os.environ["TAVILY_API_KEY"]
# except KeyError:
#     st.error("🔴 **Error:** API keys not found. Please set OPENAI_API_KEY and TAVILY_API_KEY as environment variables.")
#     st.stop()

# # --- UI & Styling ---
# st.title("🚀 Deployable Conversational AI Agent")
# st.markdown("""
# Welcome to your professional-grade AI assistant! This agent leverages advanced retrieval techniques and conversational memory to provide accurate, context-aware answers.

# **Key Features:**
# - **📚 PDF-Powered Knowledge Base:** Upload a PDF to give the agent specialized knowledge.
# - **🧠 Conversational Memory:** Ask follow-up questions and the agent will remember the context.
# - **🔍 Transparent Reasoning:** See the agent's step-by-step thought process for every answer.
# - **⚡ Real-time Streaming:** Responses are streamed token-by-token for a dynamic experience.
# """)

# # --- Agent and Tool Setup ---

# # Initialize the LLM with streaming enabled
# llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0, streaming=True)

# # Session State Initialization
# if "store" not in st.session_state:
#     st.session_state.store = {}
# if "agent_with_chat_history" not in st.session_state:
#     st.session_state.agent_with_chat_history = None
# if "retriever" not in st.session_state:
#     st.session_state.retriever = None

# # Function to get or create a chat history for a session
# def get_session_history(session_id: str):
#     if session_id not in st.session_state.store:
#         st.session_state.store[session_id] = ChatMessageHistory()
#     return st.session_state.store[session_id]

# # --- Sidebar for PDF Upload and Agent Creation ---
# with st.sidebar:
#     st.header("⚙️ Configuration")
#     st.markdown("Upload a PDF document to create a specialized knowledge base for the agent.")
    
#     uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

#     if st.button("Create Agent"):
#         if uploaded_file is not None:
#             with st.spinner("Processing PDF and creating agent... This is a one-time setup per document."):
#                 try:
#                     # 1. Load and Process the PDF
#                     pdf_reader = pypdf.PdfReader(uploaded_file)
#                     raw_text = "".join(page.extract_text() for page in pdf_reader.pages)
#                     docs = [Document(page_content=raw_text)]

#                     # 2. Refined Retriever Setup
#                     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#                     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
#                     splits = text_splitter.split_documents(docs)
#                     vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
#                     st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                    
#                     st.success("PDF processed and knowledge base created!")

#                     # 3. Create the Tools
#                     retriever_tool = create_retriever_tool(
#                         st.session_state.retriever,
#                         "pdf_knowledge_base",
#                         "MUST USE this tool to answer any questions about the specific entities, events, and data within the user's uploaded PDF document. This document contains fictional or private information (like 'Titan Dynamics', 'Lumina Innovations', 'Agri-Mind') that DOES NOT exist on the public internet. Do not use the web search tool for these specific topics.",
#                     )
#                     search_tool = TavilySearchResults(api_key=tavily_api_key)
#                     tools = [retriever_tool, search_tool]

#                     # 4. Create the Agent
#                     prompt = hub.pull("hwchase17/react-chat")

#                     # --- CRITICAL FIX for Infinite Loops ---
#                     # We modify the agent's prompt to give it an explicit instruction
#                     # on how to handle cases where a tool returns no useful information.
#                     # This prevents the agent from getting stuck in a loop.
#                     prompt.template = prompt.template + "\n\nIMPORTANT: If a tool returns no useful information or an empty result, you MUST state that the information is not available and that you cannot answer the question. Do not try to use the same tool again."
                    
#                     agent = create_react_agent(llm, tools, prompt)
#                     agent_executor = AgentExecutor(
#                         agent=agent, 
#                         tools=tools, 
#                         verbose=True, 
#                         handle_parsing_errors=True,
#                         return_intermediate_steps=True
#                     )

#                     # 5. Add Memory
#                     st.session_state.agent_with_chat_history = RunnableWithMessageHistory(
#                         agent_executor,
#                         get_session_history,
#                         input_messages_key="input",
#                         history_messages_key="chat_history",
#                     )
                    
#                     st.success("Conversational agent is ready to chat!")
#                 except Exception as e:
#                     st.error(f"An error occurred during agent creation: {e}")
#         else:
#             st.warning("Please upload a PDF file first.")

# # --- Main Chat Interface ---

# async def process_user_question(user_question):
#     intermediate_steps = []
#     full_response = ""
#     answer_container = st.empty()
    
#     with st.expander("Show agent's thought process..."):
#         steps_container = st.empty()

#     async for event in st.session_state.agent_with_chat_history.astream_events(
#         {"input": user_question},
#         config={"configurable": {"session_id": "main_chat"}},
#         version="v2"
#     ):
#         kind = event["event"]
        
#         if kind == "on_chain_start":
#             steps_container.markdown(f"**Thinking...**")

#         elif kind == "on_tool_start":
#             intermediate_steps.append(f"**Tool Used:** `{event['name']}` with input: `{event['data'].get('input')}`")
#             steps_container.markdown("\n\n".join(intermediate_steps))

#         elif kind == "on_tool_end":
#             output = event['data'].get('output')
#             if output:
#                 display_output = (output[:300] + '...') if len(output) > 300 else output
#                 intermediate_steps.append(f"**Tool Output:**\n\n```\n{display_output}\n```")
#                 steps_container.markdown("\n\n".join(intermediate_steps))
        
#         elif kind == "on_chat_model_stream":
#             content = event["data"]["chunk"].content
#             if content:
#                 full_response += content
#                 answer_container.markdown(full_response + "▌")

#     answer_container.markdown(full_response)

# st.header("💬 Chat Interface")

# chat_history = get_session_history("main_chat")
# for msg in chat_history.messages:
#     st.chat_message(msg.type).write(msg.content)

# if user_question := st.chat_input("Ask a question about your document or a general topic..."):
#     with st.chat_message("user"):
#         st.markdown(user_question)

#     if st.session_state.agent_with_chat_history is None:
#         with st.chat_message("assistant"):
#             st.warning("Please create the agent using the sidebar first.")
#     else:
#         with st.chat_message("assistant"):
#             asyncio.run(process_user_question(user_question))
# # --- Footer ---
# st.sidebar.markdown("---")
# st.sidebar.markdown("Built with ❤️ using [Streamlit](https://streamlit.io) and [LangChain](https://www.langchain.com/).")
# st.sidebar.markdown("This agent is designed to handle specific, fictional entities and data. It is not intended for real-world applications.")












# # main.py
# # ==============================================================================
# # FINAL PRODUCTION-READY AGENT
# #
# # This is the complete, deployable version of the AI agent, including all
# # features and refinements developed throughout the project.
# #
# # To run this application:
# # 1. Install all required libraries:
# #    pip install streamlit pypdf faiss-cpu langchain-openai langchain tavily-python langchainhub
# #
# # 2. Create a `requirements.txt` file with the following content:
# #    streamlit
# #    pypdf
# #    faiss-cpu
# #    langchain-openai
# #    langchain
# #    tavily-python
# #    langchainhub
# #
# # 3. Set your API keys as environment variables.
# #    - OPENAI_API_KEY="sk-..."
# #    - TAVILY_API_KEY="tvly-..."
# #
# # ==============================================================================

# # --- Core Imports ---
# import streamlit as st
# import os
# import pypdf
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.tools.retriever import create_retriever_tool
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain import hub
# from langchain.agents import create_react_agent, AgentExecutor
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain.docstore.document import Document

# # --- Asyncio Fix for Streamlit and LangChain on Windows ---
# import sys
# import asyncio
# if sys.platform == "win32":
#     try:
#         loop = asyncio.get_running_loop()
#     except RuntimeError:
#         loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(loop)

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Production AI Agent",
#     page_icon="🏆",
#     layout="wide",
# )

# # --- API Key Setup ---
# try:
#     openai_api_key = os.environ["OPENAI_API_KEY"]
#     tavily_api_key = os.environ["TAVILY_API_KEY"]
# except KeyError:
#     st.error("🔴 **Error:** API keys not found. Please set OPENAI_API_KEY and TAVILY_API_KEY as environment variables.")
#     st.stop()

# # --- UI & Styling ---
# st.title("🏆 Production-Ready Conversational AI Agent")
# st.markdown("""
# Welcome to the final version of your professional-grade AI assistant! This agent is now fully refined and ready for deployment.

# **Key Features:**
# - **📚 PDF-Powered Knowledge Base:** Upload a PDF to give the agent specialized knowledge.
# - **🧠 Conversational Memory:** Ask follow-up questions and the agent will remember the context.
# - **🔍 Transparent Reasoning:** See the agent's step-by-step thought process for every answer.
# - **⚡ Real-time Streaming:** Responses are streamed token-by-token for a dynamic experience.
# - **🛡️ Robust Logic:** Includes advanced error handling and loop prevention.
# """)

# # --- Agent and Tool Setup ---

# # Initialize the LLM with streaming enabled
# llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0, streaming=True)

# # Session State Initialization
# if "store" not in st.session_state:
#     st.session_state.store = {}
# if "agent_with_chat_history" not in st.session_state:
#     st.session_state.agent_with_chat_history = None
# if "retriever" not in st.session_state:
#     st.session_state.retriever = None

# # Function to get or create a chat history for a session
# def get_session_history(session_id: str):
#     if session_id not in st.session_state.store:
#         st.session_state.store[session_id] = ChatMessageHistory()
#     return st.session_state.store[session_id]

# # --- Sidebar for PDF Upload and Agent Creation ---
# with st.sidebar:
#     st.header("⚙️ Configuration")
#     st.markdown("Upload a PDF document to create a specialized knowledge base for the agent.")
    
#     uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

#     if st.button("Create Agent"):
#         if uploaded_file is not None:
#             with st.spinner("Processing PDF and creating agent... This is a one-time setup per document."):
#                 try:
#                     # 1. Load and Process the PDF
#                     pdf_reader = pypdf.PdfReader(uploaded_file)
#                     raw_text = "".join(page.extract_text() for page in pdf_reader.pages)
#                     docs = [Document(page_content=raw_text)]

#                     # 2. Refined Retriever Setup
#                     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#                     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
#                     splits = text_splitter.split_documents(docs)
#                     vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
#                     st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                    
#                     st.success("PDF processed and knowledge base created!")

#                     # 3. Create the Tools
#                     retriever_tool = create_retriever_tool(
#                         st.session_state.retriever,
#                         "pdf_knowledge_base",
#                         "MUST USE this tool to answer any questions about the specific entities, events, and data within the user's uploaded PDF document. This document contains fictional or private information (like 'Titan Dynamics', 'Lumina Innovations', 'Agri-Mind', 'PageRank') that DOES NOT exist on the public internet. Do not use the web search tool for these specific topics.",
#                     )
#                     search_tool = TavilySearchResults(api_key=tavily_api_key)
#                     tools = [retriever_tool, search_tool]

#                     # 4. Create the Agent
#                     prompt = hub.pull("hwchase17/react-chat")

#                     # --- FINAL REFINEMENT: More Nuanced Reasoning ---
#                     # We modify the prompt to teach the agent how to handle tool failures gracefully.
#                     # This prevents infinite loops and encourages smarter tool switching.
#                     prompt.template = """Assistant is a large language model trained by OpenAI.

# Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text in response to a wide range of prompts and questions, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

# Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

# Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

# TOOLS:
# ------

# Assistant has access to the following tools:

# {tools}

# To use a tool, please use the following format:

# ```
# Thought: Do I need to use a tool? Yes
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# Observation: the result of the action
# ```

# When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

# ```
# Thought: Do I need to use a tool? No
# Final Answer: [your response here]
# ```

# **IMPORTANT INSTRUCTIONS:**
# 1. If the `pdf_knowledge_base` tool returns no useful information, you MUST consider if the question is a general knowledge question that the `tavily_search_results_json` tool could answer.
# 2. If you have tried the relevant tool and it provided no answer, you MUST state that the information is not available and that you cannot answer the question. Do not try the same tool again for the same question.

# Begin!

# Previous conversation history:
# {chat_history}

# New input: {input}
# {agent_scratchpad}"""
                    
#                     agent = create_react_agent(llm, tools, prompt)
#                     agent_executor = AgentExecutor(
#                         agent=agent, 
#                         tools=tools, 
#                         verbose=True, 
#                         handle_parsing_errors=True,
#                         return_intermediate_steps=True
#                     )

#                     # 5. Add Memory
#                     st.session_state.agent_with_chat_history = RunnableWithMessageHistory(
#                         agent_executor,
#                         get_session_history,
#                         input_messages_key="input",
#                         history_messages_key="chat_history",
#                     )
                    
#                     st.success("Conversational agent is ready to chat!")
#                 except Exception as e:
#                     st.error(f"An error occurred during agent creation: {e}")
#         else:
#             st.warning("Please upload a PDF file first.")

# # --- Main Chat Interface ---

# async def process_user_question(user_question):
#     intermediate_steps = []
#     full_response = ""
#     answer_container = st.empty()
    
#     with st.expander("Show agent's thought process..."):
#         steps_container = st.empty()

#     async for event in st.session_state.agent_with_chat_history.astream_events(
#         {"input": user_question},
#         config={"configurable": {"session_id": "main_chat"}},
#         version="v2"
#     ):
#         kind = event["event"]
        
#         if kind == "on_chain_start":
#             steps_container.markdown(f"**Thinking...**")

#         elif kind == "on_tool_start":
#             intermediate_steps.append(f"**Tool Used:** `{event['name']}` with input: `{event['data'].get('input')}`")
#             steps_container.markdown("\n\n".join(intermediate_steps))

#         elif kind == "on_tool_end":
#             output = event['data'].get('output')
#             if output:
#                 display_output = (output[:300] + '...') if len(output) > 300 else output
#                 intermediate_steps.append(f"**Tool Output:**\n\n```\n{display_output}\n```")
#                 steps_container.markdown("\n\n".join(intermediate_steps))
        
#         elif kind == "on_chat_model_stream":
#             content = event["data"]["chunk"].content
#             if content:
#                 full_response += content
#                 answer_container.markdown(full_response + "▌")

#     answer_container.markdown(full_response)

# st.header("💬 Chat Interface")

# chat_history = get_session_history("main_chat")
# for msg in chat_history.messages:
#     st.chat_message(msg.type).write(msg.content)

# if user_question := st.chat_input("Ask a question about your document or a general topic..."):
#     with st.chat_message("user"):
#         st.markdown(user_question)

#     if st.session_state.agent_with_chat_history is None:
#         with st.chat_message("assistant"):
#             st.warning("Please create the agent using the sidebar first.")
#     else:
#         with st.chat_message("assistant"):
#             asyncio.run(process_user_question(user_question))










import streamlit as st
import os
import pdfplumber
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.docstore.document import Document
import asyncio
import datetime

# Asyncio fix for Windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Page configuration
st.set_page_config(
    page_title="Production AI Agent",
    page_icon="🏆",
    layout="wide",
)

# API key setup
try:
    openai_api_key = os.environ["OPENAI_API_KEY"]
    tavily_api_key = os.environ["TAVILY_API_KEY"]
except KeyError:
    st.error("🔴 **Error:** API keys not found. Please set OPENAI_API_KEY and TAVILY_API_KEY as environment variables.")
    st.stop()

# UI & styling
st.title("🏆 Production-Ready Conversational AI Agent")
st.markdown("""
Welcome to the final version of your professional-grade AI assistant! This agent is now fully refined and ready for deployment.

**Key Features:**
- **📚 PDF-Powered Knowledge Base:** Upload a PDF to give the agent specialized knowledge.
- **🧠 Conversational Memory:** Ask follow-up questions and the agent will remember the context.
- **🔍 Transparent Reasoning:** See the agent's step-by-step thought process for every answer.
- **⚡ Real-time Streaming:** Responses are streamed token-by-token for a dynamic experience.
- **🛡️ Robust Logic:** Includes advanced error handling and loop prevention.
""")

# Agent and tool setup
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0, streaming=True)

# Session state initialization
if "store" not in st.session_state:
    st.session_state.store = {}
if "agent_with_chat_history" not in st.session_state:
    st.session_state.agent_with_chat_history = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Function to get or create chat history
def get_session_history(session_id: str):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Sidebar for PDF upload and agent creation
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Upload a PDF document to create a specialized knowledge base for the agent.")
    
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

    if st.button("Create Agent"):
        if uploaded_file is not None:
            with st.spinner("Processing PDF and creating agent... This is a one-time setup per document."):
                try:
                    # Load and process the PDF with pdfplumber
                    with pdfplumber.open(uploaded_file) as pdf:
                        raw_text = "".join(page.extract_text() for page in pdf.pages if page.extract_text())
                    docs = [Document(page_content=raw_text)]

                    # Refined retriever setup with smaller chunks
                    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_documents(docs)
                    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
                    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                    
                    st.success("PDF processed and knowledge base created!")

                    # Create tools
                    retriever_tool = create_retriever_tool(
                        st.session_state.retriever,
                        "pdf_knowledge_base",
                        "Use this tool for questions about the PDF document. It contains specific information not found on the internet.",
                    )
                    search_tool = TavilySearchResults(api_key=tavily_api_key)
                    tools = [retriever_tool, search_tool]

                    # Create agent with simplified prompt
                    prompt = hub.pull("hwchase17/react-chat")
                    prompt.template = """You are an assistant with access to tools for answering questions. Use the tools wisely and provide accurate responses.

TOOLS:
{tools}

Use the following format:

Thought: Do I need to use a tool? Yes
Action: [tool name]
Action Input: [input]
Observation: [result]

When ready to respond:

Thought: Do I need to use a tool? No
Final Answer: [your response]

Important: If a tool doesn't provide useful information, try another tool or state that the information is unavailable.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}"""
                    
                    agent = create_react_agent(llm, tools, prompt)
                    agent_executor = AgentExecutor(
                        agent=agent, 
                        tools=tools, 
                        verbose=True, 
                        handle_parsing_errors=True,
                        return_intermediate_steps=True
                    )

                    # Add memory
                    st.session_state.agent_with_chat_history = RunnableWithMessageHistory(
                        agent_executor,
                        get_session_history,
                        input_messages_key="input",
                        history_messages_key="chat_history",
                    )
                    
                    st.success("Conversational agent is ready to chat!")
                except Exception as e:
                    st.error(f"An error occurred during agent creation: {e}")
        else:
            st.warning("Please upload a PDF file first.")

# Main chat interface
async def process_user_question(user_question):
    intermediate_steps = []
    full_response = ""
    answer_container = st.empty()
    
    with st.expander("Show agent's thought process..."):
        steps_container = st.empty()

    async for event in st.session_state.agent_with_chat_history.astream_events(
        {"input": user_question},
        config={"configurable": {"session_id": "main_chat"}},
        version="v2"
    ):
        kind = event["event"]
        
        if kind == "on_chain_start":
            steps_container.markdown(f"**Thinking...**")

        elif kind == "on_tool_start":
            intermediate_steps.append(f"**Tool Used:** `{event['name']}` with input: `{event['data'].get('input')}`")
            steps_container.markdown("\n\n".join(intermediate_steps))

        elif kind == "on_tool_end":
            output = event['data'].get('output')
            if output:
                display_output = (output[:300] + '...') if len(output) > 300 else output
                intermediate_steps.append(f"**Tool Output:**\n\n```\n{display_output}\n```")
                steps_container.markdown("\n\n".join(intermediate_steps))
        
        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                full_response += content
                answer_container.markdown(full_response + "▌")

    answer_container.markdown(full_response)

st.header("💬 Chat Interface")

chat_history = get_session_history("main_chat")
for msg in chat_history.messages:
    timestamp = datetime.datetime.now().strftime("%H:%M")
    st.chat_message(msg.type).markdown(f"**{timestamp}** - {msg.content}")

if user_question := st.chat_input("Ask a question about your document or a general topic..."):
    with st.chat_message("user"):
        timestamp = datetime.datetime.now().strftime("%H:%M")
        st.markdown(f"**{timestamp}** - {user_question}")

    if st.session_state.agent_with_chat_history is None:
        with st.chat_message("assistant"):
            st.warning("Please create the agent using the sidebar first.")
    else:
        with st.chat_message("assistant"):
            asyncio.run(process_user_question(user_question))

# footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using [Streamlit](https://streamlit.io) and [LangChain](https://www.langchain.com/).")
st.sidebar.markdown("This agent is designed to handle specific, fictional entities and data. It is not intended for real-world applications.")
