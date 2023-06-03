import os
import streamlit as st

from streamlit_chat import message as st_message
from util_funcs import chat_history_download
from core_funcs import generate_answer

# read openAI API
os.environ["OPENAI_API_Key"]=st.secrets.OPENAI_API_KEY

# history variable to store history for st.chat
if "history" not in st.session_state:
    st.session_state.history = []
#  chat history for  the chain
if "chat_his" not in st.session_state:
    st.session_state.chat_his=[]



# these columns will help with centering the title
col1,col_center,col3 = st.columns([1,2,1])
with col_center :
    st.title('Q&A Products')


# file uploading
files=st.file_uploader("Choose a file",accept_multiple_files=True,type=["pdf",'docx','txt'])

    
if files!=[]:
    # only start parsing when the user uploads a document 
   
    if st.session_state.history==[]:
            st.session_state.history.append({"message":"What would you like to know about this document?", "is_user": False})

    for chat in st.session_state.history:
        # unpacking the messages stored
            st_message(**chat) 

    if len(st.session_state.history)==1:
        # if the user didnt type anything yet , create some empty space 

        for i in range(0,10):
            st.text(" ")

    text_box=st.text_input("Talk to your pdf", key="input_text", on_change=generate_answer,args=[files])

    # centering the re-run button
    col7, col8, col_9 , col10, col11 = st.columns(5)
    with col_9 :

        rerun = st.button('Re-run',help="this button will delete your chat history and prompt you to create a new one")
        if rerun:
            # this will delete the chat history , as well as the cache to avoid any bugs
            st.session_state.history = []
            st.session_state.chat_his = []
            st.cache_data.clear()
            st.experimental_rerun()
    # centering the download history button
    col_12,col_13,col_14 =st.columns(3)
    with col_13:
        st.download_button(
                label="Download chat history",
                data=chat_history_download(st.session_state.chat_his),
                file_name='chat_history.pdf',
                mime='application/pdf',help="download chat history in PDF Format",
            )
