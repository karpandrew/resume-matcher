# 🧠 Resume Matcher AI

A Streamlit web app for recruiters and hiring teams to evaluate student resumes using AI-powered matching logic. This tool helps identify top candidates based on hard skills, contextual experience, and soft skill inference — not just keyword stuffing.

---

## 🚀 How It Works

Upload a batch of student resumes (PDF/DOCX) and paste in a job description. The app uses OpenAI's GPT model and custom logic to rank each resume against three evaluation levels:

---

### 🎯 Match Model: 3 Levels

| Level | Description | Weight |
|-------|-------------|--------|
| **Level 1: Skills Match** | Does the resume include the required hard skills or tools mentioned in the job description? Uses deterministic keyword matching for accuracy. | **60%** |
| **Level 2: Experience Relevance** | Does the candidate have project or internship experience that aligns with the *goals and intent* of the job (e.g., building tools, APIs, systems)? GPT is used to interpret project relevance, not just tool names. | **25%** |
| **Level 3: Soft Skill Inference** | Does the resume demonstrate communication, clarity, initiative, or project leadership? GPT evaluates how well projects are articulated (STAR method, metrics, portfolio links, etc.) and how the resume is structured. | **15%** |

Each level outputs:
- A score (0 to 1)
- A bullet-point rationale
- A combined weighted score used to rank all candidates

---

## 📁 Input Format

- Upload resumes as `.pdf` or `.docx`
- Paste a job description in the text box
- App will return:
  - Match Score
  - Level 1, 2, and 3 Scores + Rationale
  - Extracted email
  - Profile URL (LinkedIn preferred, GitHub as fallback)

---

## 📤 CSV Export

Each run generates a downloadable `.csv` containing:

- Resume filename
- Match Score
- Level 1 / 2 / 3 scores
- Rationale (as bullet points)
- Extracted email
- Extracted profile URL (LinkedIn > GitHub)

---

## 🛠 Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
