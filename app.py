"""
ResumeIQ - AI Resume Analyzer
Flask backend: extracts text from PDF/DOCX/TXT resumes, runs NLP-driven
heuristics to score ATS-friendliness, extracts profile details, and
generates prioritized improvement recommendations. Optionally compares
the resume against a pasted job description using TF-IDF cosine similarity.
"""

import os
import re
import tempfile
from collections import Counter

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as sk_text
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# ---------------------------------------------------------------------------
# Knowledge bases used for heuristic scoring
# ---------------------------------------------------------------------------

SKILLS_LIST = [
    # Programming
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "Ruby", "PHP", "Swift", "Kotlin", "R", "MATLAB", "Scala",
    # Web
    "HTML", "CSS", "React", "Angular", "Vue", "Node.js", "Express", "Django",
    "Flask", "Spring Boot", "jQuery", "Bootstrap", "Tailwind CSS", "REST API",
    "GraphQL",
    # Data / ML
    "Machine Learning", "Deep Learning", "Data Science", "Data Analysis",
    "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Keras",
    "NLP", "Computer Vision", "Data Visualization", "Tableau", "Power BI",
    "Excel", "SQL", "Statistics", "Big Data", "Hadoop", "Spark", "ETL",
    # Cloud / DevOps
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Jenkins", "CI/CD", "Git",
    "GitHub", "Linux", "Terraform", "Ansible",
    # Databases
    "MySQL", "PostgreSQL", "MongoDB", "Oracle", "SQLite", "Redis", "Firebase",
    # Design
    "Figma", "Adobe Photoshop", "Adobe Illustrator", "UI/UX Design", "Canva",
    # Business / Management
    "Project Management", "Agile", "Scrum", "Product Management",
    "Business Analysis", "Stakeholder Management", "Risk Management",
    "Strategic Planning", "Six Sigma",
    # Finance / Banking
    "Financial Analysis", "Accounting", "Auditing", "Taxation",
    "Investment Banking", "Financial Modeling", "Budgeting", "GST", "Tally",
    "SAP FICO", "Portfolio Management", "Credit Analysis", "KYC", "AML",
    # Marketing / Sales
    "Digital Marketing", "SEO", "SEM", "Content Marketing",
    "Social Media Marketing", "Google Analytics", "Sales", "CRM",
    "Salesforce", "Lead Generation", "Market Research",
    # HR
    "Recruitment", "Talent Acquisition", "Employee Relations", "Payroll",
    "Performance Management",
    # Soft skills
    "Leadership", "Communication", "Teamwork", "Problem Solving",
    "Critical Thinking", "Time Management", "Adaptability", "Creativity",
    "Negotiation", "Conflict Resolution", "Decision Making",
    "Public Speaking", "Collaboration", "Attention to Detail",
    "Customer Service",
]

ACTION_VERBS = [
    "achieved", "managed", "led", "developed", "designed", "implemented",
    "created", "built", "launched", "increased", "decreased", "improved",
    "optimized", "streamlined", "coordinated", "analyzed", "executed",
    "delivered", "negotiated", "spearheaded", "initiated", "established",
    "generated", "reduced", "resolved", "mentored", "trained", "supervised",
    "organized", "facilitated", "presented", "authored", "automated",
    "engineered", "deployed", "migrated", "architected", "collaborated",
    "drove", "transformed", "planned",
]

SECTION_KEYWORDS = {
    "summary": ["summary", "objective", "profile", "about me", "career objective"],
    "experience": ["experience", "employment history", "work history",
                   "professional experience", "internship"],
    "education": ["education", "academic background", "qualification"],
    "skills": ["skills", "technical skills", "core competencies", "key skills"],
    "projects": ["projects", "academic projects", "personal projects"],
    "certifications": ["certifications", "certificates", "licenses"],
    "achievements": ["achievements", "awards", "honors", "accomplishments"],
}

DEGREE_MAP = {
    "b.tech": "B.Tech", "btech": "B.Tech", "bachelor of technology": "B.Tech",
    "b.e": "B.E", "m.tech": "M.Tech", "mtech": "M.Tech",
    "mba": "MBA", "bca": "BCA", "mca": "MCA",
    "b.sc": "B.Sc", "bsc": "B.Sc", "m.sc": "M.Sc", "msc": "M.Sc",
    "phd": "Ph.D", "ph.d": "Ph.D", "diploma": "Diploma",
    "b.com": "B.Com", "bcom": "B.Com", "m.com": "M.Com", "mcom": "M.Com",
    "bachelor": "Bachelor's Degree", "master": "Master's Degree",
    "12th": "12th / HSC", "hsc": "12th / HSC",
    "10th": "10th / SSC", "ssc": "10th / SSC",
}

NAME_SKIP_WORDS = ["resume", "curriculum", "vitae", "address", "phone", "email", "objective"]


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(filepath):
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_docx(filepath):
    doc = Document(filepath)
    return "\n".join(p.text for p in doc.paragraphs)


def extract_text(filepath, ext):
    if ext == "pdf":
        return extract_text_from_pdf(filepath)
    if ext == "docx":
        return extract_text_from_docx(filepath)
    with open(filepath, "r", errors="ignore") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_contact_info(txt):
    email = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", txt)
    phone = re.search(r"(\+?\d{1,3}[-.\s]?)?(\d{10}|\d{5}[-.\s]?\d{5})", txt)
    linkedin = re.search(r"(https?://)?(www\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+", txt)
    github = re.search(r"(https?://)?(www\.)?github\.com/[A-Za-z0-9\-_]+", txt)
    return {
        "email": email.group(0) if email else None,
        "phone": phone.group(0).strip() if phone else None,
        "linkedin": linkedin.group(0) if linkedin else None,
        "github": github.group(0) if github else None,
    }


def extract_name(txt):
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    for line in lines[:6]:
        if "@" in line or re.search(r"\d{3,}", line):
            continue
        words = line.split()
        if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if w[0].isalpha()):
            if not any(kw in line.lower() for kw in NAME_SKIP_WORDS):
                return line
    return "Not Detected"


def detect_sections(txt):
    lower = txt.lower()
    found = {}
    for key, patterns in SECTION_KEYWORDS.items():
        found[key] = any(re.search(r"\b" + re.escape(p) + r"\b", lower) for p in patterns)
    return found


def extract_skills(txt):
    lower = txt.lower()
    found = []
    for skill in SKILLS_LIST:
        if re.search(r"\b" + re.escape(skill.lower()) + r"\b", lower):
            found.append(skill)
    return found


def extract_education(txt):
    lower = txt.lower()
    found = []
    for key, label in DEGREE_MAP.items():
        if re.search(r"\b" + re.escape(key) + r"\b", lower) and label not in found:
            found.append(label)
    return found


def count_action_verbs(txt):
    lower = txt.lower()
    count = 0
    for verb in ACTION_VERBS:
        count += len(re.findall(r"\b" + verb + r"\b", lower))
    return count


def count_quantifiable(txt):
    pattern = r"(\d+(\.\d+)?%|\$\d+[kKmM]?|₹\s?\d+[kKlLcC]?|\b\d{2,}\+?\b)"
    return len(re.findall(pattern, txt))


def count_bullets(txt):
    count = 0
    for line in txt.split("\n"):
        s = line.strip()
        if s.startswith(("•", "-", "*", "●", "▪", "◦")) or re.match(r"^\d+[.)]\s", s):
            count += 1
    return count


def extract_experience_years(txt):
    matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s*(?:of)?\s*experience", txt.lower())
    if matches:
        try:
            return max(float(m) for m in matches)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

def compute_score(sections_found, skills_found, education_found, contact,
                   action_verb_count, quant_count, word_count, bullet_count):
    scores = {}
    recommendations = []
    strengths = []

    # 1. Contact info — 10 pts
    contact_score = 0
    if contact["email"]:
        contact_score += 4
    else:
        recommendations.append({"priority": "High", "text": "Add a professional email address at the top of your resume."})
    if contact["phone"]:
        contact_score += 3
    else:
        recommendations.append({"priority": "High", "text": "Include a reachable contact phone number."})
    if contact["linkedin"] or contact["github"]:
        contact_score += 3
    else:
        recommendations.append({"priority": "Medium", "text": "Add a LinkedIn (or GitHub/portfolio) link to strengthen credibility."})
    if contact_score == 10:
        strengths.append("Complete contact information provided.")
    scores["contact"] = {"label": "Contact Info", "score": contact_score, "max": 10}

    # 2. Structure — 20 pts
    structure_score = 0
    for sec in ["summary", "experience", "education", "skills"]:
        if sections_found.get(sec):
            structure_score += 4
        else:
            pr = "High" if sec in ["experience", "skills"] else "Medium"
            recommendations.append({"priority": pr, "text": f'Add a clearly labeled "{sec.title()}" section to your resume.'})
    if sections_found.get("projects") or sections_found.get("certifications"):
        structure_score += 4
    else:
        recommendations.append({"priority": "Low", "text": "Consider adding a Projects or Certifications section to show initiative."})
    if structure_score >= 16:
        strengths.append("Well-organized resume with clear, standard sections.")
    scores["structure"] = {"label": "Structure", "score": structure_score, "max": 20}

    # 3. Skills — 15 pts
    skill_count = len(skills_found)
    skills_score = round(min(15, skill_count * 1.5), 1)
    if skill_count < 5:
        recommendations.append({"priority": "High", "text": "List more relevant skills — aim for at least 8-10 keywords matched to your target role."})
    else:
        strengths.append(f"Good range of relevant skills detected ({skill_count} keywords).")
    scores["skills"] = {"label": "Skills Coverage", "score": skills_score, "max": 15}

    # 4. Experience quality — 25 pts
    verb_score = min(12, action_verb_count * 1.5)
    quant_score = min(13, quant_count * 2.5)
    exp_score = round(verb_score + quant_score, 1)
    if action_verb_count < 4:
        recommendations.append({"priority": "High", "text": 'Start bullet points with strong action verbs like "Led", "Developed", or "Achieved" instead of passive phrasing.'})
    if quant_count < 3:
        recommendations.append({"priority": "High", "text": 'Quantify your achievements with numbers or metrics (e.g., "Improved efficiency by 25%").'})
    else:
        strengths.append("Resume includes measurable, quantified achievements.")
    scores["experience_quality"] = {"label": "Experience Quality", "score": exp_score, "max": 25}

    # 5. Formatting / length — 15 pts
    if 400 <= word_count <= 800:
        length_score = 8
        strengths.append("Resume length is well within the ideal range.")
    elif 250 <= word_count < 400 or 800 < word_count <= 1000:
        length_score = 5
        recommendations.append({"priority": "Medium", "text": "Adjust resume length — aim for 400-800 words for optimal readability."})
    else:
        length_score = 2
        which = "short" if word_count < 250 else "long"
        recommendations.append({"priority": "High", "text": f"Your resume seems too {which} ({word_count} words). Aim for 400-800 words."})
    bullet_score = 7 if bullet_count >= 8 else round(bullet_count * 0.85, 1)
    if bullet_count < 8:
        recommendations.append({"priority": "Medium", "text": "Use bullet points consistently to describe responsibilities and achievements for easy scanning."})
    format_score = round(length_score + bullet_score, 1)
    scores["formatting"] = {"label": "Formatting & Length", "score": format_score, "max": 15}

    # 6. Education — 15 pts
    education_score = 15 if education_found else 0
    if not education_found:
        recommendations.append({"priority": "Medium", "text": "Clearly mention your educational qualifications with degree, institution, and year."})
    else:
        strengths.append("Education details are clearly mentioned.")
    scores["education"] = {"label": "Education", "score": education_score, "max": 15}

    total = round(sum(v["score"] for v in scores.values()), 1)

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    recommendations.sort(key=lambda r: priority_order.get(r["priority"], 3))

    return scores, total, recommendations, strengths


def compute_jd_match(resume_text, jd_text):
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf = vectorizer.fit_transform([resume_text, jd_text])
        similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except ValueError:
        similarity = 0.0
    match_pct = round(similarity * 100, 1)

    words = re.findall(r"[A-Za-z][A-Za-z+#]{2,}", jd_text)
    stop = sk_text.ENGLISH_STOP_WORDS
    words = [w.lower() for w in words if w.lower() not in stop and len(w) > 3]
    freq = Counter(words)
    top_keywords = [w for w, _ in freq.most_common(30)]

    resume_lower = resume_text.lower()
    matched = [w for w in top_keywords if w in resume_lower][:15]
    missing = [w for w in top_keywords if w not in resume_lower][:15]

    return {"match_percentage": match_pct, "matched_keywords": matched, "missing_keywords": missing}


def score_label(total):
    if total >= 85:
        return "Excellent"
    if total >= 70:
        return "Good"
    if total >= 50:
        return "Needs Improvement"
    return "Poor"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def analyze_resume(text, jd_text, filename):
    contact = extract_contact_info(text)
    name = extract_name(text)
    sections_found = detect_sections(text)
    skills_found = extract_skills(text)
    education_found = extract_education(text)
    action_verb_count = count_action_verbs(text)
    quant_count = count_quantifiable(text)
    word_count = len(text.split())
    bullet_count = count_bullets(text)
    exp_years = extract_experience_years(text)

    scores, total, recommendations, strengths = compute_score(
        sections_found, skills_found, education_found, contact,
        action_verb_count, quant_count, word_count, bullet_count,
    )

    result = {
        "filename": filename,
        "name": name,
        "contact": contact,
        "word_count": word_count,
        "bullet_count": bullet_count,
        "experience_years": exp_years,
        "sections_found": sections_found,
        "skills_found": skills_found,
        "education_found": education_found,
        "action_verb_count": action_verb_count,
        "quant_count": quant_count,
        "scores": scores,
        "total_score": total,
        "score_label": score_label(total),
        "recommendations": recommendations,
        "strengths": strengths,
    }

    if jd_text and len(jd_text.strip()) > 20:
        result["jd_match"] = compute_jd_match(text, jd_text)

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["resume"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file. Please upload a PDF, DOCX, or TXT file."}), 400

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({"error": "File too large. Maximum size is 5MB."}), 400

    jd_text = request.form.get("job_description", "").strip()
    ext = file.filename.rsplit(".", 1)[1].lower()

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + ext) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        text = extract_text(tmp_path, ext)
    except Exception:
        return jsonify({"error": "Could not read this file. It may be corrupted or password-protected."}), 400
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not text or len(text.strip()) < 30:
        return jsonify({"error": "Could not extract readable text from this file. If it's a scanned/image PDF, try a text-based version instead."}), 400

    result = analyze_resume(text, jd_text, file.filename)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
