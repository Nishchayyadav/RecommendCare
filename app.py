from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from userInputPred import get_user_input
import re
import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

def generate_query(data):
    gender = "Male" if data[0] == 1 else "Female"
    age = f"{data[1]} years old"
    smoker_status = "a smoker" if data[3] == 1 else "a non-smoker"
    cigs_per_day = f"smoking {data[4]} cigarettes per day" if data[3] == 1 else ""
    bp_meds = "is on blood pressure medication" if data[5] == 1 else "is not on blood pressure medication"
    stroke_history = "has a history of stroke" if data[6] == 1 else "has no history of stroke"
    hypertension = "has hypertension" if data[7] == 1 else "does not have hypertension"
    diabetes = "has diabetes" if data[8] == 1 else "does not have diabetes"

    if data[9] < 120:
        sys_bp = "normal systolic blood pressure"
    elif data[9] < 140:
        sys_bp = "elevated systolic blood pressure"
    else:
        sys_bp = "high systolic blood pressure"

    if data[10] < 80:
        dia_bp = "normal diastolic blood pressure"
    elif data[10] < 90:
        dia_bp = "elevated diastolic blood pressure"
    else:
        dia_bp = "high diastolic blood pressure"

    if data[11] < 18.5:
        bmi = "underweight"
    elif data[11] < 24.9:
        bmi = "normal weight"
    elif data[11] < 29.9:
        bmi = "overweight"
    else:
        bmi = "obese"

    if data[12] < 60:
        heart_rate = "a low heart rate (below 60 bpm, bradycardia)"
    elif 60 <= data[12] <= 100:
        heart_rate = "a normal heart rate (60-100 bpm)"
    else:
        heart_rate = "a high heart rate (above 100 bpm, tachycardia)"

    glucose = data[13];

    query = f"The patient is {gender}, {age}. "

    if data[3] == 1:
        query += f"They are {smoker_status} and {cigs_per_day}. "
    if data[5] == 1:
        query += f"The patient {bp_meds}. "
    if data[6] == 1:
        query += f"The patient {stroke_history}. "
    if data[7] == 1:
        query += f"The patient {hypertension} with {sys_bp}, {dia_bp}. "
    if data[8] == 1:
        query += f"The patient {diabetes} with {glucose}. "
    if data[12] < 60:
        query += f"The patient has {heart_rate}. "
    if data[12] > 100:
        query += f"The patient has {heart_rate}. "
    if data[11] >= 30:
        query += f"The patient is classified as {bmi}. "
    query += "What recommendations can you provide to improve their cardiovascular health?"

    return query

def remove_markdown(text):
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(---|\*\*\*|___)\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()
    text = re.sub(r"\n", " ", text)
    return text

loader = TextLoader("ragSource.txt")
documents = loader.load()
text = documents[0].page_content
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n", ".", " "]
)

chunks = splitter.split_text(text)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

documents = [Document(page_content=chunk) for chunk in chunks]
# print(f"len {len(documents)}")
docsearch = FAISS.from_documents(documents, embeddings)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


prompt_template = """
You are a helpful assistant trained in cardiovascular health management.
Provide a short, direct answer to the query based on the context without anyhthing else.

Context:
{context}

Question:
{question}

Answer:
"""

llm = HuggingFaceHub(
    repo_id="google/gemma-1.1-7b-it",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model_kwargs={"temperature": 0.6, "max_length": 1024, "return_full_text" : False}
)

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "query"]
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

def main(current_smoker, cigs_per_day, bpm, prevalent_stroke, prevalent_hyp, diabetes):
    # patient_data = [1,58,1,20,0,0,1,0,102,57,29.01,90,50]
    # patient_data.insert(2, None)    
    patient_data = get_user_input("data.csv", current_smoker, cigs_per_day, bpm, prevalent_stroke, prevalent_hyp, diabetes)
    query = generate_query(patient_data)
    response = qa.invoke({"query": query})
    print(f"{remove_markdown(response['result'])}")
    return remove_markdown(response['result'])
    # print(f"Source Documents: {response['source_documents']}")
    
# main()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Extract data from form
    name = request.form.get('name')
    current_smoker = int(request.form.get('smoker'))
    cigs_per_day = float(request.form.get('cigs_per_day') or 0)  # Default to 0 if no input
    bpm = int(request.form.get('bp_meds'))
    prevalent_stroke = int(request.form.get('stroke'))
    prevalent_hyp = int(request.form.get('hypertension'))
    diabetes = int(request.form.get('diabetes'))

    result = main(current_smoker, cigs_per_day, bpm, prevalent_stroke, prevalent_hyp, diabetes)

    return jsonify({
        'status': 'success',
        'result': result
    })

if __name__ == '__main__':
    app.run(debug=True)