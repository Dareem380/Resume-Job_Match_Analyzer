# AI Resume–Job Match Analyzer

Find out how well your resume fits a job description, which skills you are missing, and what to improve, in seconds.

This tool compares a resume with a job description using TF-IDF text similarity and a skill-matching engine. It then prints a clear report and saves a visual chart.

---

## Problem Statement

- Job seekers often don't know why their resume gets rejected.
- Many companies use automated screening that filters resumes by keywords.
- Students and freshers lack quick feedback on how well they fit a role.

## Our Solution

An analyzer that reads your resume and the job description and returns:

- An **overall match score** (0–100%)
- The skills you **already have** and the skills you **are missing**
- A **skill-category breakdown** chart
- **Suggestions** on what to add to your resume

---

## Features

| Feature | Description |
|---|---|
| **Match score** | A weighted blend of skill coverage (60%) and TF-IDF cosine similarity (40%) |
| **Skill detection** | Detects 40+ skills across programming, data and AI, web and cloud, databases, and soft skills |
| **Gap analysis** | Lists missing skills and suggests how to cover them |
| **Visual report** | Score donut and skill-coverage bar chart, saved as PNG |
| **Works offline** | No API keys and no internet needed |
| **Your own files** | Analyse any resume and job description saved as `.txt` |

---

## Getting Started

### Prerequisites
- Python 3.8 or newer

### Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install scikit-learn matplotlib
```

### Usage

**Run with the built-in sample data:**

```bash
python resume_matcher.py
```

**Run with your own files:**

```bash
python resume_matcher.py --resume my_resume.txt --job job_description.txt
```

### Example output

```
========================================================
  OVERALL MATCH SCORE : 41.3%
  Skill match         : 50.0%
  Text similarity     : 14.1%
========================================================
  Partial match - tailor your resume and learn the key gaps.

  Skills matched : python, sql, data analysis, machine learning, numpy, pandas, git, ...
  Skills missing : data visualization, deep learning, nlp, pytorch, tensorflow, aws, ...
```

A chart named `resume_match_report.png` is saved next to the script.

---

## How It Works

1. **Clean:** the resume and job description are lowercased and normalised.
2. **Text similarity:** both are converted to TF-IDF vectors (unigrams and bigrams) and compared with cosine similarity.
3. **Skill matching:** a built-in skill bank is searched in both texts, and the skills the job needs are compared with the skills the resume has.
4. **Scoring:** `final score = 0.6 × skill coverage + 0.4 × text similarity` (similarity is scaled up, since short texts naturally score low).
5. **Report:** the score, verdict, matched and missing skills, suggestions and chart are produced.

### Verdict levels

| Score | Verdict |
|---|---|
| 75% and above | Excellent match |
| 55% to 74% | Good match |
| 35% to 54% | Partial match |
| Below 35% | Low match |

---

## Project Structure

```
├── resume_matcher.py        # Main script
├── resume_match_report.png  # Generated chart (created on run)
└── README.md
```

## Customising Skills

Open `resume_matcher.py` and edit the `SKILLS` dictionary to add skills for any field, such as finance, design, or healthcare:

```python
SKILLS = {
    "Design": ["figma", "photoshop", "ux research"],
    ...
}
```

---

## Future Scope

- Upload PDF and DOCX resumes directly
- Web app with Streamlit or Flask
- Semantic matching with transformer embeddings (for example, "ML" and "machine learning")
- Auto-rewrite suggestions with an LLM
- Job-board integration to rank many openings against one resume

## Tech Stack

Python · scikit-learn · Matplotlib
