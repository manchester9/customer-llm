import openai
import os
import uuid
import json
import streamlit as st
from google.cloud import dialogflow_v2 as dialogflow
from my_secrets import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

# Set up Dialogflow credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "service_key.json"
client = dialogflow.SessionsClient()

# Load project ID from service key file
with open("service_key.json") as f:
    service_key = json.load(f)
project_id = service_key["project_id"]

# Generate unique session ID for current user
session_id = str(uuid.uuid4())
session_path = client.session_path(project_id, session_id)

# Define Streamlit app
def chatbot():
    # Set page title
    st.set_page_config(page_title="Chatbot Demo")

    # Define page layout
    st.title("Welcome to the Chatbot Demo")
    st.write("Enter your message below:")

    # Define user input field
    user_input = st.text_input(label="User Input")

    # Process user input and display chatbot response
    if user_input:
        # Call GPT model
        gpt_response = openai.Completion.create(engine="text-davinci-002", prompt=user_input, max_tokens=50)
        text_input = dialogflow.types.TextInput(text=gpt_response["choices"][0]["text"], language_code="en-US")
        query_input = dialogflow.types.QueryInput(text=text_input)
        response = client.detect_intent(session=session_path, query_input=query_input)

        st.text(response.query_result.fulfillment_text)

if __name__ == "__main__":
    chatbot()
