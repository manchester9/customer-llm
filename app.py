import streamlit as st

def main():
    st.title("Welcome to LLM powered applications")
    st.markdown("---")
    st.write("Please select from the drop-down what you would like to do today and then click Submit.")
    
    option = st.selectbox("Select an option", ("Document QNA", "Summarization"))
    st.markdown("---")
    
    if st.button("Submit"):
        if option == "Document QNA":
            st.write("Redirecting to Document QNA page...")
        elif option == "Summarization":
            st.write("Redirecting to Summarization page...")

if __name__ == "__main__":
    main()

