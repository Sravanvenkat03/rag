from flask import Flask, request, jsonify
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import requests
import fitz  # PyMuPDF for PDF text extraction
import os
import spacy
import re

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")

search_endpoint = "https://triconsearch3.search.windows.net"
search_api_key = "TmdhvcAzupEAq3EsJqfWpkwhbCwfuz0GUa3ADwoE8tAzSeDqblxl"
index_name = "searchindex"

API_KEY = 'f34c8f7c947948ff83c462f15d091b6f'
ENDPOINT = 'https://azurechatbottricon.openai.azure.com/'
DEPLOYMENT_NAME = 'gpt-35-turbo'

search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=AzureKeyCredential(search_api_key))

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

def chunk_text(text, chunk_size=100, overlap=10):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def upload_documents(documents):
    try:
        result = search_client.upload_documents(documents=documents)
        for doc_result in result:
            doc_id = getattr(doc_result, 'key', 'N/A')
            status_code = getattr(doc_result, 'status_code', 'N/A')
            error_message = getattr(doc_result, 'error_message', 'None')
            print(f"Document ID: {doc_id}, Status: {status_code}, Error: {error_message}")
    except Exception as e:
        print(f"Error uploading documents: {e}")

def get_openai_response(prompt):
    try:
        headers = {
            'Content-Type': 'application/json',
            'api-key': API_KEY
        }
        data = {
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.7
        }
        response = requests.post(
            f"{ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/completions?api-version=2023-05-15",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Error getting OpenAI response: {e}")
        return None

def extract_keywords(query):
    cleaned_query = re.sub(r'[^\w\s]', '', query)
    doc = nlp(cleaned_query)
    keywords = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
    keywords += [ent.text.lower() for ent in doc.ents]
    return keywords

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    file = request.files.get('pdf')
    if not file:
        return jsonify({"error": "No file provided"}), 400
    
    pdf_path = os.path.join("pdf_files", file.filename)
    file.save(pdf_path)
    
    extracted_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(extracted_text)
    
    documents = [{"id": str(i), "content": chunk} for i, chunk in enumerate(chunks)]
    
    upload_documents(documents)
    
    return jsonify({"message": "PDF processed and documents uploaded successfully"})

@app.route('/ask', methods=['POST'])
def ask():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        keywords = extract_keywords(query)
        search_text = " OR ".join(keywords)

        search_results = search_client.search(search_text=search_text, top=5)
        results_list = list(search_results)

        search_results_content = [result['content'] for result in results_list]

        if not search_results_content:
            return jsonify({"error": "No relevant content found"}), 400

        pdf_matches = [content for content in search_results_content if any(keyword in content.lower() for keyword in keywords)]

        if pdf_matches:
            context = "\n".join(pdf_matches)
            prompt = f"Here are some search results:\n{context}\n\nPlease summarize the information relevant to the following question with points: {query}"
            openai_response = get_openai_response(prompt)
            return jsonify({"results": pdf_matches, "openai_response": openai_response})
        else:
            return jsonify({"error": "No relevant content found"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    if not os.path.exists("pdf_files"):
        os.makedirs("pdf_files")
    app.run(debug=True)
