import streamlit as st
import boto3
# import pandas as pd
import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt
from botocore.exceptions import ClientError
import json
from datetime import datetime
from dotenv import load_dotenv
import os, re, io
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
st.set_page_config(layout="wide")

from utils.clients import get_s3_client

## utils imports
from utils.clients import get_session, get_bedrock_client
from utils.data_utils import load_data
from utils.document_utils import read_document, get_metadata, generate_s2t_mapping
from utils.er_diagram_utils import get_er_json, extract_json_from_string, create_er_diagram, fallback_visualization
from utils.visualizations_utils import table_selection_component, display_visualizations #, display_cross_table_correlation
from utils.chatbot_utils import generate_conversation #, chatbot_response
from utils.correlation_utils import correlation_utils_main
from utils.ai_visualization_utils import get_relevant_visualizations, streamlit_process_visualizations
from utils.multiattribute_analysis_utils import multi_attribute_analysis_main
from DynamicDataManipulation.data.data_processing import upload_file, preview_data

import invoke_agent as agenthelper
# Initialize session state variables
if 'document_uploaded' not in st.session_state:
    st.session_state.document_uploaded = False
if 's2t_mapping' not in st.session_state:
    st.session_state.s2t_mapping = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'er_diagram_generated' not in st.session_state:
    st.session_state.er_diagram_generated = False

# Load the data
transactions_df, loan_applications_df, accounts_df, customers_df, customer_feedback_df, account_alerts_df,modified_df = load_data()

# Display Financial Dashboard
dfs = {
    'Transactions': transactions_df,
    'Loan Applications': loan_applications_df,
    'Accounts': accounts_df,
    'Customers': customers_df,
    'Customer Feedback': customer_feedback_df,
    'Account Alerts': account_alerts_df,
    'Modified table': modified_df
}


# Main Page
def main_page():

    # Select visualization mode
    visualization_mode = st.radio("Select Visualization Mode", ["Regular Visualizations", "Cross-Table Correlation", "Multi-Attribute visualization"])

    if visualization_mode == "Regular Visualizations":
        selected_tables, selected_columns = table_selection_component(dfs)
        display_visualizations(selected_tables, selected_columns, dfs)
    elif visualization_mode == "Cross-Table Correlation":
        correlation_utils_main(dfs)
    else:
    # selected_tables, selected_columns = table_selection_component(dfs)
    # display_cross_table_correlation(selected_tables, selected_columns, dfs)
        multi_attribute_analysis_main(dfs)

    ####################################
    # Sidebar - Upload Requirement Document
    st.sidebar.title("Upload Requirement Document")
    uploaded_file = st.sidebar.file_uploader("Upload a requirement document (PDF)", type=['pdf'])

    if uploaded_file is not None:
        st.session_state.document_uploaded = True
        document_text = read_document(uploaded_file)
        st.sidebar.success("Document uploaded successfully!")
        metadata = get_metadata()
        
        # Generate visualizations 
        if st.sidebar.button("Generate AI Visualizations"):
            with st.spinner("Generating AI-suggested visualizations..."):
                logger.info("Starting AI visualization generation process")
                bedrock_runtime = get_bedrock_client()
                visualizations = get_relevant_visualizations(document_text, metadata, dfs ,bedrock_runtime)
                logger.info(f"Received {len(visualizations)} visualization suggestions from AI")

                st.subheader("AI-suggested Visualizations(Work in Progress)")
                streamlit_process_visualizations(visualizations, metadata, dfs, bedrock_runtime)

        # Generate S2T Mapping Button
        if st.sidebar.button("Generate S2T Mapping"):
            with st.spinner("Generating S2T Mapping..."):
                s2t_mapping = generate_s2t_mapping(document_text, metadata)
                st.session_state.s2t_mapping = s2t_mapping
                st.sidebar.success("S2T Mapping Generated!")
                st.title("Source-to-Target (S2T) Mapping")
                st.markdown(st.session_state.s2t_mapping)

        # Display the "Generate ER Diagram" button
        if st.sidebar.button("Generate ER Diagram"):
            print("Button pressed for ER diagram")
            with st.spinner("Generating ER Diagram..."):
                print("Entered generating ER  diagram")
                json_file_path = './schema.json'

                print("Generating ER diagram..")
                if st.session_state.s2t_mapping is None:
                    with st.spinner("Generating S2T Mapping..."):
                        st.session_state.s2t_mapping = generate_s2t_mapping(document_text, metadata)
                        # st.title("Source-to-Target (S2T) Mapping")
                        # st.markdown(st.session_state.s2t_mapping)

                er_json = get_er_json(document_text, st.session_state.s2t_mapping, metadata)
                result = extract_json_from_string(er_json)

                if result:
                    json_string = json.dumps(result, indent=4)
                    with open(json_file_path, 'w') as json_file:
                        json_file.write(json_string)

                    print(f"JSON schema saved to {json_file_path}")

                    output_image_path = 'er_diagram.png'
                    diagram_path = create_er_diagram(json_file_path, output_image_path)
                    if diagram_path is None:
                        fallback_visualization(json_file_path)
                        st.image('fallback_er_diagram.png',caption="ER Diagram (Fallback Visualization)", use_column_width=True)
                    else:
                        from PIL import Image
                        img = Image.open(output_image_path)
                        # st.image(img, caption="ER Diagram", use_column_width=True)

                    st.success("ER Diagram Generated!")
                    st.session_state.er_diagram_generated = True
                else:
                    st.write(f"There was an error with the ER diagram image generation, refer to the below structure for the diagram \n {er_json}")

    # Display ER Diagram and S2T Mapping even after asking a question
    if st.session_state.er_diagram_generated and st.session_state.s2t_mapping:
        st.title("Source-to-Target (S2T) Mapping")
        st.markdown(st.session_state.s2t_mapping)

        output_image_path = 'er_diagram.png'
        if os.path.exists(output_image_path):
            from PIL import Image
            img = Image.open(output_image_path)
            st.image(img, caption="ER Diagram", use_column_width=True)
        else:
            st.image('fallback_er_diagram.png', caption="ER Diagram (Fallback Visualization)", use_column_width=True)


    def clarify_intent(text, metadata):
        prompt = f"""
        I want to know if the following user query can be answered using just the database information, metadata and tables and columns and relations mapping info, OR will it require the actual data in the tables to answer the query.
Be very strict and go to the data only for questions that will necessarily require data to answer the questions or make visualizations using info etc.
Classify the following user query as either 'metadata' or 'data'. 
Respond with one word only no quotes: 'metadata' or 'data'.
    {text}

Metadata:
{metadata}
"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                ],
            }
        ]
        
        bedrock_runtime = get_bedrock_client()
        response = bedrock_runtime.converse(
            modelId=MODEL_ID,
            messages=messages,
        )
        logger.info("Intent Classifier Output ->  %s", response)
        return response["output"]["message"]["content"][0]["text"]


    # Chatbot Interface
    st.subheader("Assistant")
    user_input = st.text_input("Ask a question about the Database or the Requirements Document and S2T mapping:")

    if user_input:
        metadata = get_metadata()
        intent = clarify_intent(user_input, metadata)
        if intent.strip().lower() == 'metadata':
            context = f"""
            Requirement Document:
              {document_text}
            S2T Mapping:
            {st.session_state.s2t_mapping}
            Metadata:
            {metadata}
            
            """
            system_prompts = [{"text": f"""You are a business analyst helper that can answer queries about the database and
            requirement documents or related questions to help the tech BA. You have the below context, to answer the BA's questions.
            Context:
            {context}

            You are a bot that will answer questions relating to the metadata, if some operation occurs you're supposed to tell us what are the columns/tables/relations that will be affected by it and
     what will be the effect on the entire database."""}]
            user_message = {
                "role": "user",
                "content": [{"text": f"User Question: {user_input}"}]
            }
            messages = [user_message]
            try:
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                with st.spinner("Generating response..."):
                    response = generate_conversation(
                        bedrock_client, model_id="anthropic.claude-3-sonnet-20240229-v1:0", 
                        system_prompts=system_prompts, messages=messages
                    )
                output_message = response['output']['message']
                st.session_state.chat_history.append(("AI", output_message['content'][0]['text']))
                st.session_state.chat_history.append(("You", user_input))

            except ClientError as err:
                logger.error("A client error occurred: %s", err.response['Error']['Message'])
        
        # else
        else:
           # session_agent = boto3.Session(
           #     aws_access_key_id=aws_access_key_id,
           #     aws_secret_access_key=aws_secret_access_key,
           #     region_name='us-east-1'
           # )
            session_agent = get_session()
            event = {
                "sessionId": "ICEW0SH701",
                "question": user_input
            }
            response = agenthelper.lambda_handler(event, session_agent, None)
            try:
                if response and 'body' in response and response['body']:
                    response_data = json.loads(response['body'])
                    logger.info("Response Body Keys:%s ",list(response.keys()))
            #        logger.info("TRACE & RESPONSE DATA ->  %s", response_data)
                    try:
                        output_message = response_data['trace_data']
                        st.session_state.chat_history.append(("AI", output_message))
                        st.session_state.chat_history.append(("You", user_input))
                    except:
                        st.error("Internal Error from Athena, please try again. Try changing the phrasing of your query as well. ")
                        output_message = "Apologies, but an error occurred. Please rerun the application"
                else:
                    st.error("Invalid or empty response received")
                    output_message = "Apologies, but an error occurred. Please rerun the application"
            except json.JSONDecodeError as e:
                st.error(f"JSON decoding error: {str(e)}")
                output_message = "Apologies, but an error occurred. Please rerun the application"


    # Display chat history
    for role, message in reversed(st.session_state.chat_history):
        st.write(f"**{role}:** {message}")

    # Footer
    st.sidebar.write("Made by Kidon Team 4 - Arceus")

def data_manipulation_page():
    if 'uploaded_file' not in st.session_state:
        st.session_state['df'] = upload_file()
        if st.session_state['df'] is not None:
            st.rerun()

    if 'uploaded_file' in st.session_state:
        df = preview_data(st.session_state['df'])
        
    if 'df' in st.session_state and st.session_state['df'] is not None:
        if st.button("Save to S3"):
            print("'Save to S3' button clicked")
            # Save the modified DataFrame to S3
            file_name = "new.csv"
            with st.spinner(f"Saving DataFrame to S3 with file name: {file_name}..."):
                try:
                    save_to_s3(df, file_name)
                    print("DataFrame saved to S3 successfully")
                    st.success(f"DataFrame saved to S3 with the file name: {file_name}")

                    # Add the saved DataFrame to the dfs dictionary
                    dfs['Modified DataFrame - ' + file_name] = st.session_state['df']
                    print("DataFrame added to dfs dictionary")
                    st.success(f"DataFrame added to dfs for visualization with the key: 'Modified DataFrame - {file_name}'")
                except ClientError as e:
                    print(f"ClientError occurred: {e}")

def save_to_s3(df, file_name):
    print("Inside save_to_s3 function")
    # Create an S3 client
    s3_client = get_s3_client()

    # Specify the bucket name and file path
    bucket_name = 'kidonteam4-final'
    file_path = f'ModifiedDatasets/{file_name}'

    # Save the DataFrame to a CSV file
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Upload the CSV file to S3
    try:
        print(f"Uploading file to S3 bucket: {bucket_name}, file path: {file_path}")
        s3_client.put_object(Body=csv_buffer.getvalue().encode('utf-8'), Bucket=bucket_name, Key=file_path)
        print("File uploaded to S3 successfully")
    except ClientError as e:
        print(f"ClientError occurred: {e}")
        raise e


def invoke_bedrock_model(model_id, prompt):
    bedrock_runtime = get_bedrock_client()
    response = bedrock_runtime.invoke_model(
        modelId=model_id,
        body=json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": 1000,
            "temperature": 0.7,
            "top_p": 1,
            "stop_sequences": []
        })
    )
    
    return json.loads(response['body'].read())['completion']

def generate_conversation(bedrock_client, model_id, system_prompts, messages):
    temperature = 0.1
    top_k = 200
    inference_config = {"temperature": temperature}
    additional_model_fields = {"top_k": top_k}

    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields
    )

    token_usage = response['usage']
    return response

def generate_dataframe_modif_code(df, objective):
    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    prompt = [{"text":f'''Human: As a proficient Python developer, your task is to design a script that meticulously customizes a DataFrame to meet specific user-defined criteria.

    Key requirements for the script include:
    1. Simplicity and Clarity: Keep the code streamlined, while carefully considering the types of values within the DataFrame.
2. Handling Missing Values: Efficiently manage any missing (NA) values in the DataFrame.
3. Precise Customization: Accurately implement the modifications requested by the user.
4. Final Output: Ensure the script returns the altered DataFrame.

Enclose your solution in a function named modify_dataframe(df). This function should be complete, self-contained, and executable as provided. Deliver the solution in Python code.

Make certain your implementation precisely conforms to the specifications, ensuring accuracy in every aspect of the user's request, such as manipulating specific columns or hand>

Here are the first 10 rows of the dataframe: {df.head(10)}. Here is the instruction: {objective}.  make sure you just give the Python code enclosed in ```python and nothing else'''}]

    bedrock_runtime = get_bedrock_client()
    
    user_message = {
            "role": "user",
            "content": [{"text": f"User Question: {objective}"}]
        }
    messages = [user_message]

    response = generate_conversation(bedrock_runtime,model_id, prompt,messages)
    completion = response['output']['message']['content'][0]['text']
    python_match = re.search(r'```python\n(.*?)```', completion, re.DOTALL)

    if python_match:
        python_string = python_match.group(1)

    return python_string

def run_app():
    st.title("metailyst")
    st.sidebar.title("Navigation")
    pages = {
        "Business Analyst Tool": main_page,
        "Data Manipulation": data_manipulation_page,
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    if selection == "Business Analyst Tool":
        main_page()
    elif selection == "Data Manipulation":
        data_manipulation_page()

if __name__ == "__main__":
    run_app()