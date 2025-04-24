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

def get_embedding(text: str) -> List[float]:
    response = openai.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding

def compute_similarity(job_emb: List[float], res_emb: List[List[float]]) -> List[float]:
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity([job_emb], res_emb)[0]

def extract_email(text: str) -> str:
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else "Not found"

def extract_profile_url(text: str) -> str:
    urls = re.findall(r"https?://[^\s\)\]]+", text)
    linkedin = next((url for url in urls if "linkedin.com/in" in url), None)
    github = next((url for url in urls if "github.com" in url), None)
    return linkedin or github or ""

def match_skills(text: str, keywords: List[str]) -> float:
    text_lower = text.lower()
    matches = [kw for kw in keywords if kw.lower() in text_lower]
    return len(matches) / len(keywords) if keywords else 0

def get_level_scores(job_description: str, resume_text: str) -> tuple:
    skill_keywords = ["python", "javascript", "html", "css", "sql", "java", "c++", "react", "node.js", "git", "linux"]
    level_1_score = match_skills(resume_text, skill_keywords)
    level_1_reason = f"- Matched {int(level_1_score * len(skill_keywords))} of {len(skill_keywords)} core skills."

    prompt = (
        "Evaluate this resume against the job description across:\n"
        "- Level 2 (Contextual Experience): What is the job actually trying to accomplish? "
        "Think about the role's goals — like building a chatbot, integrating with Slack, or developing a full-stack system. "
        "Has the candidate done real work that maps to those goals? Focus on implementation and project relevance. "
        "Ignore tool names and keywords — evaluate based on the depth and relevance of the experience to the job’s intent.\n"
        "- Level 3 (Soft Skills): Inferred from how the resume is written and structured. Consider the clarity and articulation of project descriptions, use of the STAR method, quantitative results, and the number and depth of personal or academic projects. "
        "Since these are students, focus on communication ability, initiative, and whether the resume demonstrates thoughtfulness, organization, and effort.\n\n"
        f"Job Description:\n{job_description[:1000]}\n\n"
        f"Resume:\n{resume_text[:1000]}\n\n"
        "Return a JSON object with the following keys:\n"
        '{"level_2_score": float, "level_2_reason": str, '
        '"level_3_score": float, "level_3_reason": str}'
    )

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    content = response.choices[0].message.content.strip()
    try:
        result = eval(content) if content.startswith("{") else {}
        level_2 = float(result.get("level_2_score", 0))
        level_3 = float(result.get("level_3_score", 0))
        return (
            level_1_score, level_2, level_3,
            level_1_reason.strip(),
            result.get("level_2_reason", "").strip(),
            result.get("level_3_reason", "").strip()
        )
    except Exception:
        return level_1_score, 0.0, 0.0, level_1_reason, "N/A", "N/A"

# --- UI ---

st.title("📄 Resume Matcher AI")

job_description = st.text_area("Paste the job description here", height=200)

uploaded_files = st.file_uploader("Upload resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Run Matching") and job_description and uploaded_files:
    with st.spinner("Processing..."):
        job_emb = get_embedding(job_description)

        resume_texts = []
        resume_names = []
        for file in uploaded_files:
            try:
                text = extract_text_from_pdf(file) if file.name.endswith(".pdf") else extract_text_from_docx(file)
                resume_texts.append(text)
                resume_names.append(file.name)
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

        results = []
        for name, text in zip(resume_names, resume_texts):
            email = extract_email(text)
            url = extract_profile_url(text)
            level_1, level_2, level_3, r1, r2, r3 = get_level_scores(job_description, text)
            weighted_score = 0.5 * level_1 + 0.35 * level_2 + 0.15 * level_3
            results.append((name, weighted_score, level_1, level_2, level_3, r1, r2, r3, email, url))

        results.sort(key=lambda x: x[1], reverse=True)
        st.success("Matching complete!")

        st.subheader("📋 Top Matches")
        for tup in results:
            name, score, level_1, level_2, level_3, r1, r2, r3, email, url = tup
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
                        <summary style="color: #888; font-size: 12px;">ℹ️ Level 1 (Skills): {level_1:.2f}</summary>
                        <div style="color: #ccc; font-size: 12px; margin-top: 4px;">
                            {r1}
                        </div>
                    </details>
                    <details style="margin-top: 6px;">
                        <summary style="color: #888; font-size: 12px;">ℹ️ Level 2 (Experience): {level_2:.2f}</summary>
                        <div style="color: #ccc; font-size: 12px; margin-top: 4px;">
                            {r2}
                        </div>
                    </details>
                    <details style="margin-top: 6px;">
                        <summary style="color: #888; font-size: 12px;">ℹ️ Level 3 (Soft Skills): {level_3:.2f}</summary>
                        <div style="color: #ccc; font-size: 12px; margin-top: 4px;">
                            {r3}
                        </div>
                    </details>
                </div>
            """, unsafe_allow_html=True)

        # CSV Export
        df = pd.DataFrame(results, columns=[
            "Resume Name", "Match Score",
            "Level 1 Score", "Level 2 Score", "Level 3 Score",
            "Level 1 Rationale", "Level 2 Rationale", "Level 3 Rationale",
            "Email", "Profile URL"
        ])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download results as CSV", csv, "resume_matches.csv", "text/csv")