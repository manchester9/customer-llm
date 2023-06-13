import streamlit as st

from streamlit.components.v1 import html
from style import nav_page


# save the uploaded file
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files=[]


st.header('Welcome to LLM powered applications')
st.write("")
st.markdown('')

#st.header(st.session_state.title)
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font">Select from the dropdown menu below what you would like to do today and click submit</p>', unsafe_allow_html=True)
option = st.selectbox(
    "",['DocumentQ&a', 'summary'])

col1,col2,col3=st.columns([2,1,2])
with col2:
    submit=st.button("submit")
if submit and option=='DocumentQ&a':
    nav_page("qa")
elif submit and option=='summary':
    nav_page('summary')
   

