import streamlit as st
import logging
from datetime import datetime, timedelta

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Constants
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Streamlit page configuration
st.set_page_config(layout="wide")

# Utility imports from clients.py
from utils.clients import get_session, get_bedrock_client

# AWS Clients
session = get_session()  # Get a session with credentials
logs_client = session.client('logs')  # CloudWatch Logs client
bedrock_client = get_bedrock_client()  # Bedrock client


def fetch_lambda_logs(log_group_name, start_time=None, end_time=None):
    """
    Fetch logs for a given Lambda function from CloudWatch Logs.
    """
    if not start_time:
        start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)  # Last 24 hours
    if not end_time:
        end_time = int(datetime.now().timestamp() * 1000)  # Now

    logs = []
    next_token = None

    while True:
        params = {
            "logGroupName": log_group_name,
            "startTime": start_time,
            "endTime": end_time,
        }
        if next_token:
            params["nextToken"] = next_token  # Include nextToken only if it's not None

        response = logs_client.filter_log_events(**params)

        for event in response.get('events', []):
            logs.append(event['message'])

        next_token = response.get('nextToken', None)
        if not next_token:
            break

    return logs


def answer_question_with_bedrock(log_text, question):
    """
    Use Bedrock to answer a question based on logs.
    """
    # Enhanced prompt
    s2t_prompt = f"""
    You are an intelligent assistant designed to analyze AWS Lambda logs. 
    Below are logs from a Lambda function, which include request IDs, queries, execution times, errors, and other events.

    Logs:
    {log_text}

    Based on the logs, please analyze and answer the following question:
    {question}

    Provide a concise, accurate, and human-readable response.
    """

    messages = [
        {
            "role": "user",
            "content": [{"text": s2t_prompt}]
        }
    ]

    response = bedrock_client.converse(
        modelId=MODEL_ID,
        messages=messages
    )

    return response["output"]["message"]["content"][0]["text"]


# Streamlit App
def main():
    st.title("Lambda Logs Q&A with AWS Bedrock")
    st.write("Fetch logs of a Lambda function and ask questions about them.")

    # Initialize session state for logs and answer
    if "logs_text" not in st.session_state:
        st.session_state.logs_text = None
    if "answer" not in st.session_state:
        st.session_state.answer = None

    # Pre-fill the log group name for the Lambda function
    log_group_name = st.text_input("Enter Lambda Log Group Name", value="/aws/lambda/kidon4final")
    fetch_logs = st.button("Fetch Logs")

    if fetch_logs and log_group_name.strip():
        with st.spinner("Fetching logs..."):
            try:
                logs = fetch_lambda_logs(log_group_name)
                if logs:
                    st.success("Logs fetched successfully!")
                    st.session_state.logs_text = "\n".join(logs)  # Save logs in session state
                else:
                    st.warning("No logs found for the given Lambda function.")
            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")
                st.error(f"An error occurred: {str(e)}")

    # Display logs if available
    if st.session_state.logs_text:
        st.text_area("Logs", st.session_state.logs_text, height=300)

        # Question and answer section
        question = st.text_input("Ask a question about the logs")
        get_answer = st.button("Get Answer")

        if get_answer and question.strip():
            with st.spinner("Getting the answer..."):
                try:
                    st.session_state.answer = answer_question_with_bedrock(st.session_state.logs_text, question)
                except Exception as e:
                    logger.error(f"An error occurred: {str(e)}")
                    st.error(f"An error occurred: {str(e)}")

        # Display the answer if available
        if st.session_state.answer:
            st.write("Answer:", st.session_state.answer)


if __name__ == "__main__":
    main()