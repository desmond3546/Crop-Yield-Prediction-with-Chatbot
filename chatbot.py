# Import necessary libraries
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import streamlit as st

def main():
    # Load environment variables from a .env file
    load_dotenv()

    # Load the Google API key from the environment variable
    if os.getenv("GOOGLE_API_KEY") is None or os.getenv("GOOGLE_API_KEY") == "":
        print("GOOGLE_API_KEY is not set")
        exit(1)
    else:
        print("GOOGLE_API_KEY is set")

    # Set the page configuration for the Streamlit app
    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    # File uploader for CSV files
    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        # Create a CSV agent using Google Generative AI
        agent = create_csv_agent(
            ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3),
            csv_file,
            verbose=True,
            allow_dangerous_code=True
        )

        # Input field for user questions
        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            # Display a spinner while the agent processes the question
            with st.spinner(text="In progress..."):
                # Display the agent's response
                st.write(agent.run(user_question))

if __name__ == "__main__":
    # Run the main function to start the Streamlit app
    main()
