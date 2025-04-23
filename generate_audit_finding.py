import requests
import pandas as pd
import traceback
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env into the environment

token = os.environ.get("HF_TOKEN")
TOKEN = "Bearer " + token
headers = {"Authorization": TOKEN}
base_llm_model_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

def query(payload):
    response = requests.post(base_llm_model_URL, headers=headers, json=payload)
    return response.json()

# 1. Load your CSV data (replace 'your_data.csv' with your file path)
def prepare_data():
    
    df = pd.read_csv('Audit Observations-Findings.csv')
    df = df.dropna()
    
    # Assuming observations are in column 2 and findings are in column 3 (index 0,1,2..)
    observations = df.iloc[:, 1].tolist()
    findings = df.iloc[:, 2].tolist()
    
    examples = []
    
    for ii in range(0,23):
        text = {"observation": observations[ii], "finding": findings[ii]}
        examples = examples + [text]
    
    
    # Format the examples into a string
    example_string = ""
    for example in examples:
        example_string += f"Observation: {example['observation']}\nFinding: {example['finding']}\n\n"

    return example_string

def return_audit_findings(input_observation,example_string):

    # Your question         
    question = f"Observation: {input_observation}\n\nProvide me the audit findings?"
    
    # Create the prompt
    prompt = f"I am giving few observations and their corresponding findings. Based on it,\
     for my new observations,provide findings. :\n\n{example_string}\n{question}"
    
    # Send the prompt to the model
    output = query({
        "inputs": prompt
    })    
    
    return output

if __name__ == "__main__":
    example_string = prepare_data()

    input_observation = "Subject ID 1023 received 100 mg of IMP instead of the protocol-specified 150 mg on Day 2. \
     Subject ID 45 received IMP 6 days later than the protocol-defined ±2-day administration window."
    
    output_observation = return_audit_findings(input_observation,example_string)
    print(output_observation)
