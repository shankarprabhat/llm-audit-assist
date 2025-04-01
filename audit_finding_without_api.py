import requests
import pandas as pd
import traceback
import os

token = os.environ.get("HF_TOKEN")

TOKEN = "Bearer "+token
# API_URL = 'https://api-inference.huggingface.co/models/deepset/roberta-base-squad2'

# API_URL = "https://api-inference.huggingface.co/models/google/flan-ul2" # or flan-t5-large

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large" # or flan-t5-large
# API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base" # or flan-t5-large
# API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-small" # or flan-t5-large

headers = {"Authorization": TOKEN}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# 1. Load your CSV data (replace 'your_data.csv' with your file path)

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

# Your question

observe = "Subject ID 1023 received 100 mg of IMP instead of the protocol-specified 150 mg on Day 2. \
 Subject ID 45 received IMP 6 days later than the protocol-defined ±2-day administration window."

     
question = f"Observation: {observe}\n\nProvide me the audit findings?"

# Create the prompt
prompt = f"You are an Auditor for Clinical Trials and refer to ICH GCP as well as FRS document. \
 I am giving few observations and their corresponding findings. Based on it, for my new observations, provide audit findings. :\n\n{example_string}\n{question}"

# Send the prompt to the model
output = query({
    "inputs": prompt,
})

print(output)