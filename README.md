# ResumeIQ — AI Resume Analyzer

An ML/NLP-powered resume analyzer with a polished, animated UI. Upload a
resume (PDF, DOCX, or TXT) and get:

- **ATS Compatibility Score** (0–100) with an animated circular gauge
- **Category breakdown** — Contact Info, Structure, Skills Coverage,
  Experience Quality, Formatting & Length, Education
- **Extracted details** — name, email, phone, LinkedIn, word count, bullet
  count, years of experience mentioned
- **Skills & education detection** against a curated keyword library
  spanning tech, business, finance/banking, marketing, and soft skills
- **Strengths** and **prioritized recommendations** (High/Medium/Low)
- **Optional Job Description match** — paste a JD to get a TF-IDF based
  match percentage plus matched/missing keyword chips
- A cinematic multi-step **animated loading screen** while it analyzes
- A **Download Report** button (print-to-PDF friendly styling)

## 1. Setup

```bash
cd resume-analyzer
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Run

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## 3. How it works

- `app.py` — Flask backend. Extracts raw text from the uploaded file
  (`pdfplumber` for PDFs, `python-docx` for Word files), then runs a set
  of regex/heuristic NLP extractors:
  - Contact info (email, phone, LinkedIn, GitHub) via regex
  - Section detection (Summary, Experience, Education, Skills, Projects,
    Certifications, Achievements) via keyword header matching
  - Skill extraction against a ~130-term library
  - Education/degree extraction
  - Action-verb and quantifiable-achievement counting (for judging bullet
    quality — e.g. "increased revenue by 20%" vs. vague statements)
  - Word/bullet counts for formatting checks
  - Optional JD matching via `scikit-learn`'s `TfidfVectorizer` + cosine
    similarity, plus keyword-frequency diffing for missing terms
  - A weighted scoring model (Contact 10, Structure 20, Skills 15,
    Experience Quality 25, Formatting 15, Education 15 = 100 total) that
    also generates the strengths/recommendations lists
- `templates/index.html` / `static/css/style.css` / `static/js/script.js`
  — the frontend: drag-and-drop upload, an animated multi-step loading
  sequence, and a results dashboard built from the JSON the backend
  returns.

## 4. Notes

- Max upload size: 5MB. Supported formats: `.pdf`, `.docx`, `.txt`.
- Scanned/image-only PDFs won't extract text — use a text-based export.
- All scoring is heuristic (rule-based NLP), not a trained ML classifier —
  it's designed to be transparent and explainable rather than a black box,
  which is generally what you want from an ATS-style checker.
- Feel free to tune `SKILLS_LIST`, `ACTION_VERBS`, and the weights inside
  `compute_score()` in `app.py` to match a specific industry or role.
