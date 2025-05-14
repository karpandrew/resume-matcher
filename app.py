import streamlit as st
import fitz  # PyMuPDF
import docx
import openai
import os
import re
from io import BytesIO
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- UTILS ---

def extract_text_from_pdf(file: BytesIO) -> str:
    doc = fitz.open(stream=file, filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx(file: BytesIO) -> str:
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_email(text: str) -> str:
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Not found"

def extract_profile_url(text: str) -> str:
    urls = re.findall(r"https?://[^\s\)\]]+", text)
    linkedin = next((url for url in urls if "linkedin.com/in" in url), None)
    github = next((url for url in urls if "github.com" in url), None)
    return linkedin or github or ""

def extract_structured_fields(role: str, mode: str) -> str:
    prompt = (
        f"Extract the {mode} section from this {'resume' if mode != 'job' else 'job description'}.\n"
        f"Return a clean summary suitable for semantic comparison.\n\n"
        f"{'Resume' if mode != 'job' else 'Job Description'}:\n{role[:2000]}"
    )
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def embed_text(text: str) -> List[float]:
    return openai.embeddings.create(input=text, model="text-embedding-ada-002").data[0].embedding

def get_cosine_similarity(a: List[float], b: List[float]) -> float:
    return cosine_similarity([a], [b])[0][0]

# --- UI ---

st.title("📄 Resume Matcher AI")

job_description = st.text_area("Paste the job description here", height=200)

uploaded_files = st.file_uploader("Upload resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

st.subheader("⚖️ Weight Configuration")
w1 = st.slider("🔑 Level 1: Skills Match", 0.0, 1.0, 0.5)
w2 = st.slider("💼 Level 2: Experience Match", 0.0, 1.0, 0.3)
w3 = st.slider("✨ Level 3: Soft Skills Inference", 0.0, 1.0, 0.2)
total = w1 + w2 + w3
w1, w2, w3 = w1 / total, w2 / total, w3 / total  # normalize to sum to 1

if st.button("Run Matching") and job_description and uploaded_files:
    with st.spinner("Processing..."):
        resume_texts = []
        resume_names = []
        for file in uploaded_files:
            try:
                file_bytes = BytesIO(file.read())
                file.seek(0)  # reset for future reads if needed
                text = extract_text_from_pdf(file_bytes) if file.name.endswith(".pdf") else extract_text_from_docx(file_bytes)
                resume_texts.append(text)
                resume_names.append(file.name)
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

        jd_keywords = extract_structured_fields(job_description, "keywords")
        jd_experience = extract_structured_fields(job_description, "experience")
        jd_soft = extract_structured_fields(job_description, "soft skills")

        jd_kw_emb = embed_text(jd_keywords)
        jd_exp_emb = embed_text(jd_experience)
        jd_soft_emb = embed_text(jd_soft)

        results = []

        for file, name, text in zip(uploaded_files, resume_names, resume_texts):
            email = extract_email(text)
            url = extract_profile_url(text)

            r_keywords = extract_structured_fields(text, "keywords")
            r_experience = extract_structured_fields(text, "experience")
            r_soft = extract_structured_fields(text, "soft skills")

            kw_score = get_cosine_similarity(embed_text(r_keywords), jd_kw_emb)
            exp_score = get_cosine_similarity(embed_text(r_experience), jd_exp_emb)
            soft_score = get_cosine_similarity(embed_text(r_soft), jd_soft_emb)
            final_score = w1 * kw_score + w2 * exp_score + w3 * soft_score

            results.append((name, final_score, kw_score, exp_score, soft_score, email, url))

        results.sort(key=lambda x: x[1], reverse=True)
        st.success("Matching complete!")

        st.subheader("📋 Top Matches")
        for tup in results:
            name, score, level_1, level_2, level_3, email, url = tup
            score_color = "#27ae60" if score > 0.85 else "#e67e22" if score > 0.7 else "#c0392b"
            st.markdown(f"""
                <div style="padding: 15px; border-radius: 8px; background-color: #1e1e1e; margin-bottom: 12px; border: 1px solid #333;">
                    <h4 style="margin-bottom: 5px; color: #ffffff;">{name}</h4>
                    <p style="margin: 0; color: #ffffff;">
                        <strong>Match Score:</strong> <span style="color:{score_color}; font-weight:bold;">{score:.2f}</span><br>
                        <strong>Email:</strong> {email}<br>
                        <strong>Profile URL:</strong> {url if url else "N/A"}
                    </p>
                    <details style="margin-top: 6px;">
                        <summary style="color: #888; font-size: 12px;">ℹ️ Level 1 (Keywords): {level_1:.2f}</summary>
                    </details>
                    <details style="margin-top: 6px;">
                        <summary style="color: #888; font-size: 12px;">ℹ️ Level 2 (Experience): {level_2:.2f}</summary>
                    </details>
                    <details style="margin-top: 6px;">
                        <summary style="color: #888; font-size: 12px;">ℹ️ Level 3 (Soft Skills): {level_3:.2f}</summary>
                    </details>
                </div>
            """, unsafe_allow_html=True)

        # CSV Export
        df = pd.DataFrame(results, columns=[
            "Resume Name", "Match Score",
            "Level 1: Keywords", "Level 2: Experience", "Level 3: Inference",
            "Email", "Profile URL"
        ])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download results as CSV", csv, "resume_matches.csv", "text/csv")
