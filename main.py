import getpass
import numpy as np
import openai
import os
import pandas as pd
import requests
import re
from openai import OpenAI
from pdfminer.high_level import extract_text
from singlestoredb import create_engine
from sqlalchemy import text

from llama_index import SimpleDirectoryReader
from llama_index import Document
from llama_index import VectorStoreIndex
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.embeddings import HuggingFaceEmbedding

client = openai.OpenAI()


def read_pdf_with_context():
    documents = SimpleDirectoryReader(
        input_files=["./IT_roles_description.pdf"]
    ).load_data()

    document = Document(text="\n\n".join([doc.text for doc in documents]))

    return document


def print_pdf_text(url=None, file_path=None):
    # Determine the source of the PDF (URL or local file)
    if url:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        temp_file_path = "temp_pdf_file.pdf"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(response.content)  # Save the PDF to a temporary file
        pdf_source = temp_file_path
    elif file_path:
        pdf_source = file_path  # Set the source to the provided local file path
    else:
        raise ValueError("Either url or file_path must be provided.")

    # Extract text using pdfminer
    text = extract_text(pdf_source)

    # Remove special characters except "@", "+", ".", and "/"
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s@+./:,]", "", text)

    # Format the text for better readability
    cleaned_text = cleaned_text.replace("\n\n", " ").replace("\n", " ")
    # If a temporary file was used, delete it
    if url and os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    return cleaned_text


def pinfo_extractor(resume_text):
    context = f"Resume text: {resume_text}"
    question = """ From above candidate's resume text, extract the only following details:
                Name: (Find the candidate's full name. If not available, specify "not available.")
                Email: (Locate the candidate's email address. If not available, specify "not available.")
                Phone Number: (Identify the candidate's phone number. If not found, specify "not available.")
                Years of Experience: (If not explicitly mentioned, calculate the years of experience by analyzing the time durations at each company or position listed. Sum up the total durations to estimate the years of experience. If not determinable, write "not available.")
                Skills Set: Extract the skills which are purely technical and represent them as: [skill1, skill2,... <other skills from resume>]. If no skills are provided, state "not available."
                Technologies Set: Extract the technologies which are purely technical and represent them as: [technology1, technology2,... <other technologies from resume>]. If no technologies are provided, state "not available."
                Profile: (Identify the candidate's job profile or designation. If not mentioned, specify "not available.")
                Summary: provide a brief summary of the candidate's profile without using more than one newline to segregate sections.
                """

    prompt = f"""
        Based on the below given candidate information, only answer asked question:
        {context}
        Question: {question}
    """
    # print(prompt)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful HR recruiter."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=700,
        temperature=0.5,
        n=1,  # assuming you want one generation per document
    )
    # Extract the generated response
    response_text = response.choices[
        0
    ].message.content  # response['choices'][0]['message']['content']
    # print(response_text)
    # Split the response_text into lines
    lines = response_text.strip().split("\n")

    # Now, split each line on the colon to separate the labels from the values
    # Extract the values
    name = lines[0].split(": ")[1]
    email = lines[1].split(": ")[1]
    phone_no = lines[2].split(": ")[1]
    years_of_experience = lines[3].split(": ")[1]
    skills = lines[4].split(": ")[1]
    technologies = lines[5].split(": ")[1]
    profile = lines[6].split(": ")[1]
    summary = lines[7].split(": ")[1]
    data_dict = {
        "name": name,
        "email": email,
        "phone_no": phone_no,
        "years_of_experience": years_of_experience,
        "skills": skills,
        "technologies": technologies,
        "profile": profile,
        "summary": summary,
    }
    print(data_dict, "\n")
    return data_dict


def add_data_to_db(input_dict):
    # Create the SQLAlchemy engine
    engine = create_engine("mysql://user_name:password@localhost:3306/resume_evaluator")

    # Create the SQL query for inserting the data
    query_sql = f"""
        INSERT INTO resumes_profile_data (names, email, phone_no, years_of_experience, skills, technologies, profile_name, resume_summary)
        VALUES ("{input_dict['name']}", "{input_dict['email']}", "{input_dict['phone_no']}", "{input_dict['years_of_experience']}",
        "{input_dict['skills']}", "{input_dict['technologies']}", "{input_dict['profile']}", "{input_dict['summary']}");
    """
    with engine.connect() as connection:
        connection.execute(text(query_sql))
        connection.commit()
    # print("\nData Written to resumes_profile_data table")


def search_resumes(query):
    query_sql = f"""
            SELECT names, skills, technologies, resume_summary FROM resumes_profile_data;
    """
    # print(query_sql, "\n")
    engine = create_engine("mysql://user_name:password@localhost:3306/resume_evaluator")
    connection = engine.connect()
    result = connection.execute(text(query_sql)).fetchall()
    connection.close()
    engine.dispose()
    return result


def evaluate_candidates(query):
    result = search_resumes(query)
    responses = []  # List to store responses for each candidate
    for resume_str in result:
        name = resume_str[0]
        skills = f"{resume_str[1]}"
        skills = skills.replace("[", "")
        skills = skills.replace("]", "")
        technologies = f"{resume_str[2]}"
        technologies = technologies.replace("[", "")
        technologies = technologies.replace("]", "")
        context = f"Resume text: {resume_str[3]}"
        question = f"What percentage of the job requirements does the candidate meet for the following job description? answer in 3 lines only and be effcient while answering: {query}."
        prompt = f"""
            Read below candidate information about the candidate:
            {context} And, the next skills: {skills}. And, next technologies: {technologies}
            Question: {question}
        """

        # print(f"prompt = {prompt}")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a expert HR analyst and recruiter.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.2,
            n=1,  # assuming you want one generation per document
        )
        # Extract the generated response
        response_text = response.choices[
            0
        ].message.content  # response['choices'][0]['message']['content']
        responses.append(
            (name, response_text)
        )  # Append the name and response_text to the responses list
    return responses


# def evaluate_candidates_with_basic_RAG_pipeline(query):
#     llm = OpenAI(model="gpt-3.5-turbo", temperature=0.2)
#     document = read_pdf_with_context()
#     # loads BAAI/bge-small-en-v1.5
#     embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
#     service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
#     index = VectorStoreIndex.from_documents([document], service_context=service_context)
#     query_engine = index.as_query_engine()

#     result = search_resumes(query)
#     responses = []  # List to store responses for each candidate
#     for resume_str in result:
#         name = resume_str[0]
#         context = f"{resume_str[1]}"
#         question = f"What percentage of the job requirements does the candidate meet for the following job description? answer in 3 lines only and be effcient while answering: {query}."
#         prompt = f"""
#             Read below candidate information about the candidate:
#             {context}
#             Question: {question}
#         """

#         # print(f"prompt = {prompt}")
#         response = query_engine.query(prompt)

#         # Extract the generated response
#         response_text = str(response)
#         responses.append(
#             (name, response_text)
#         )  # Append the name and response_text to the responses list
#     return responses


# def test_embeddings_score():
#     resp = openai.embeddings.create(
#         input=["feline friends say", "pio pio"], model="text-similarity-davinci-001"
#     )

#     embedding_a = resp.data[0].embedding
#     embedding_b = resp.data[1].embedding

#     similarity_score = np.dot(embedding_a, embedding_b)
#     print(f"similarity_score = {similarity_score}")


def clean_database():
    query_sql = f"""
            DELETE FROM resumes_profile_data;
    """

    engine = create_engine("mysql://user_name:password@localhost:3306/resume_evaluator")
    connection = engine.connect()
    connection.execute(text(query_sql))
    connection.close()
    engine.dispose()


def main():
    # CVs to evaluate
    file_paths = [
        "/home/jangu/Downloads/Julio_Angulo_-_Research_Engineer (3).pdf",
        "/home/jangu/Downloads/Roy Jackson - Back End Developer - Resume.pdf",
        "/home/jangu/Downloads/recruiter-resume-example.pdf",
        "/home/jangu/Downloads/Ricardo Limon Softserve Resume - Test.pdf",
    ]

    # Resume extractor
    for file_path in file_paths:
        resume_text = print_pdf_text(file_path=file_path).replace("\n", " ")
        # print("Resume Text extracted\n")
        ip_data_dict = pinfo_extractor(resume_text)
        # print("Information extracted\n")
        add_data_to_db(ip_data_dict)
        # print("\n")

    # Job description for a ML Engineer
    job_description = "Machine Learning Engineer with experience in tensorflow, keras, and tensorboard."

    # Job description for a IT Recruiter
    # job_description = "Recruiter with experience in full cycle recruiting"

    # Job description for a Back-End Developer
    # job_description = "Back-End developer with experience in Angular, SQL, and AWS."

    print(f"job_description = {job_description}")

    responses = evaluate_candidates(job_description)
    for response in responses:
        print(response)
        print("\n")

    # Clean db
    clean_database()


if __name__ == "__main__":
    main()
