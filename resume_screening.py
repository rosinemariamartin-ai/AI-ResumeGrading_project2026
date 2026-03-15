import os
import PyPDF2
import docx2txt
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Load transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# 1. Extract Text
# -------------------------------

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        text = ""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return text

    elif file_path.endswith(".docx"):
        return docx2txt.process(file_path)

    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    return ""


# -------------------------------
# 2. Preprocess Text
# -------------------------------

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and not token.is_punct
    ]
    return " ".join(tokens)


# -------------------------------
# 3. Skill List
# -------------------------------

SKILL_LIST = [
    "python", "java", "machine learning", "nlp",
    "data science", "deep learning", "sql",
    "aws", "docker", "flask", "react"
]

def extract_skills(text):
    found_skills = []
    for skill in SKILL_LIST:
        if skill in text.lower():
            found_skills.append(skill)
    return found_skills


# -------------------------------
# 4. Process Resumes
# -------------------------------

def process_resumes(resume_folder, job_description):

    job_processed = preprocess_text(job_description)
    job_embedding = model.encode([job_processed])
    job_skills = extract_skills(job_description)

    results = []

    for file in os.listdir(resume_folder):

        file_path = os.path.join(resume_folder, file)
        resume_text = extract_text(file_path)

        if not resume_text:
            continue

        processed_text = preprocess_text(resume_text)
        resume_embedding = model.encode([processed_text])

        similarity = cosine_similarity(
            job_embedding,
            resume_embedding
        )[0][0]

        resume_skills = extract_skills(resume_text)

        matching_skills = list(set(resume_skills) & set(job_skills))
        missing_skills = list(set(job_skills) - set(resume_skills))

        # Skill match ratio
        if len(job_skills) > 0:
            skill_match_ratio = len(matching_skills) / len(job_skills)
        else:
            skill_match_ratio = 0

        # Final rating formula
        final_score = (0.7 * similarity) + (0.3 * skill_match_ratio)

        rating_out_of_10 = round(final_score * 10, 2)

        results.append({
            "candidate": file,
            "similarity": round(float(similarity), 4),
            "skill_match_percent": round(skill_match_ratio * 100, 2),
            "rating": rating_out_of_10,
            "matching_skills": matching_skills,
            "missing_skills": missing_skills
        })

    ranked = sorted(results, key=lambda x: x["rating"], reverse=True)

    return ranked


# -------------------------------
# 5. Main
# -------------------------------

if __name__ == "__main__":

    job_description = """
    We are looking for a Machine Learning Engineer
    with experience in Python, NLP, AWS, and Deep Learning.
    """

    resume_folder = "resumes"

    ranked_candidates = process_resumes(resume_folder, job_description)

    print("\n===== Candidate Ranking (Out of 10) =====\n")

    for i, candidate in enumerate(ranked_candidates, 1):
        print(f"Rank {i}: {candidate['candidate']}")
        print(f"Similarity Score: {candidate['similarity']}")
        print(f"Skill Match: {candidate['skill_match_percent']}%")
        print(f"Final Rating: {candidate['rating']} / 10")
        print(f"Matching Skills: {candidate['matching_skills']}")
        print(f"Missing Skills: {candidate['missing_skills']}")
        print("-" * 50)