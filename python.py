import PyPDF2
import pandas as pd
from transformers import pipeline
import torch
import tensorflow as tf

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Step 2: Summarize using Hugging Face's API
def summarize_text_with_hf(text, hf_pipeline):
    summary = hf_pipeline(text[:1024])[0]['summary_text']  # limit text to first 1024 tokens for summarization
    return summary

# Step 3: Organize the summaries into an Excel file
def organize_into_excel(summary_dict, excel_file):
    df = pd.DataFrame.from_dict(summary_dict, orient='index', columns=['Summary'])
    df.to_excel(excel_file, index=True)

# Example usage
pdf_file = "/Users/vyro/Desktop/mypdf.pdf"
text = extract_text_from_pdf(pdf_file)

# Initialize Hugging Face summarization pipeline
hf_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Assuming you manually divide text into sections or automate this
sections = text.split("\n\n")  # This splits based on paragraph breaks, adjust as needed
summary_dict = {}
for i, section in enumerate(sections):
    summary = summarize_text_with_hf(section, hf_pipeline)
    summary_dict[f'Section {i+1}'] = summary

organize_into_excel(summary_dict, "CV_Summary.xlsx")
