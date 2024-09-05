import spacy
from transformers import pipeline
import openai
from langchain_ollama import ChatOllama
import json
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize summarization pipeline
from langchain_ollama import ChatOllama

# Initialize the ChatOllama model
summarizer = ChatOllama(
    model="mathstral:7b-v0.1-q6_K",
    temperature=0.2,
    max_tokens=512,
    top_p=0.5,
)

# Initialize the ChatOllama model
chat_model = ChatOllama(
    model="mathstral:7b-v0.1-q6_K",  # You can change this to any available model
    temperature=0.3,
    max_tokens=512,
)

def summarize_text(text):
    """
    Summarizes the given text using the ChatOllama model.

    :param text: The text to summarize.
    :return: Summarized text.
    """
    try:
        prompt = "Summarize the following text in a concise manner:"
        prompt = prompt + text
        
        # Invoke the model with the input text
        completion = summarizer.invoke(prompt)
        summary = completion.content
        print(summary)
        # The response is already a string, so we can return it directly
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Summary not available."
    
    
# Phase 2: NLP for Information Extraction and Summarization
def extract_key_information(text):
    """
    Extracts key information such as entities, methodologies, and results from text using spaCy.

    :param text: The text to analyze.
    :return: Dictionary of extracted entities and key information.
    """
    doc = nlp(text)
    entities = {
        "METHODS": [],
        "RESULTS": [],
        "DISEASES": [],
        "TREATMENTS": [],
        "OTHER": []
    }
    
    # Classify sentences into categories based on keywords
    for sent in doc.sents:
        if "method" in sent.text.lower() or "procedure" in sent.text.lower():
            entities["METHODS"].append(sent.text)
        elif "result" in sent.text.lower() or "finding" in sent.text.lower():
            entities["RESULTS"].append(sent.text)
        else:
            entities["OTHER"].append(sent.text)

    # Extract specific entities like diseases and treatments
    for ent in doc.ents:
        if ent.label_ == "DISEASE":
            entities["DISEASES"].append(ent.text)
        elif ent.label_ == "TREATMENT":
            entities["TREATMENTS"].append(ent.text)
    
    return entities



def generate_description(methods, results):
    """
    Generates a natural language description of methodologies and results using ChatOllama.

    :param methods: List of method descriptions.
    :param results: List of result descriptions.
    :return: Generated description as a string.
    """
    prompt = (
        f"Based on the following methods and results, generate a concise and clear description:\n\n"
        f"Methods:\n{methods}\n\nResults:\n{results}\n\n"
        "Provide a comprehensive summary that explains the significance and implications of these findings."
    )
    
    try:
        # Call ChatOllama to generate the description
        completion = chat_model.invoke(prompt)
        response = completion.content
        print(response)
        # The response is already a string, so we can return it directly
        return response
    except Exception as e:
        print(f"Error generating description: {e}")
        return "Description generation failed."
    
    
    
def process_and_save_article(article):
    # Extract key information
    extracted_info = extract_key_information(article['abstract'])
    
    # Generate summary
    summary = summarize_text(article['abstract'])
    
    # Generate description
    description = generate_description(extracted_info['METHODS'], extracted_info['RESULTS'])
    
    # Combine all information
    processed_article = {
        'title': article['title'],
        'abstract': article['abstract'],
        'summary': summary,
        'extracted_info': extracted_info,
        'generated_description': description
    }
    
    # Save to a file (you could also save to a database)
    with open(f"processed_articles/{article['id']}.json", 'w') as f:
        json.dump(processed_article, f, indent=2)

