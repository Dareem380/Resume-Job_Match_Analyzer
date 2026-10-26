"""
AI Resume - Job Match Analyzer
------------------------------
Compares a resume with a job description and shows:
  * Overall match score (TF-IDF + cosine similarity)
  * Skills you already have / skills you are missing
  * Skill-category breakdown chart  (saved as PNG next to this script)
  * Suggestions to improve the resume

Install:   pip install scikit-learn matplotlib
Run:       python resume_matcher.py                       -> runs with built-in sample
           python resume_matcher.py --resume my_resume.txt --job job_description.txt
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ skill bank
SKILLS = {
    "Programming": ["python", "java", "javascript", "c++", "sql", "typescript", "r", "go"],
    "Data & AI": ["machine learning", "deep learning", "nlp", "computer vision", "pandas",
                  "numpy", "scikit-learn", "tensorflow", "pytorch", "data analysis",
                  "data visualization", "statistics"],
    "Web & Cloud": ["html", "css", "react", "node.js", "flask", "django", "rest api",
                    "aws", "azure", "docker", "kubernetes", "git"],
    "Databases": ["mysql", "postgresql", "mongodb", "sqlite", "redis"],
    "Soft Skills": ["communication", "teamwork", "leadership", "problem solving",
                    "time management", "presentation"],
}

SAMPLE_RESUME = """
Priya Sharma - Final year B.Tech (CSE) student
Skills: Python, SQL, pandas, numpy, scikit-learn, machine learning, data analysis,
HTML, CSS, JavaScript, Git, MySQL, teamwork, communication.
Projects: Built a Kolam pattern generator using computer vision and OpenCV.
Created a student performance predictor with machine learning and Flask.
Hackathon finalist; presented the project to a panel of judges.
"""

SAMPLE_JOB = """
Data Scientist Intern
We are looking for a candidate with strong Python and SQL skills.
Responsibilities: build machine learning and deep learning models, perform data analysis,
create data visualization dashboards, work with pandas, numpy, tensorflow or pytorch.
Experience with NLP, docker, AWS and REST API development is a plus.
Good problem solving, communication and teamwork skills are required. Knowledge of git.
"""


# ------------------------------------------------------------------ helpers
def clean(text):
    return re.sub(r"\s+", " ", text.lower())


def find_skills(text):
    text = clean(text)
    found = {}
    for cat, items in SKILLS.items():
        hits = [s for s in items if re.search(r"(?<![\w+#.])" + re.escape(s) + r"(?![\w+#])", text)]
        found[cat] = hits
    return found


def similarity(resume, job):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    m = vec.fit_transform([clean(resume), clean(job)])
    return float(cosine_similarity(m[0], m[1])[0][0]) * 100


def analyse(resume, job):
    r_sk, j_sk = find_skills(resume), find_skills(job)
    matched, missing, breakdown = [], [], {}
    for cat in SKILLS:
        need = set(j_sk[cat])
        have = set(r_sk[cat])
        matched += sorted(need & have)
        missing += sorted(need - have)
        if need:
            breakdown[cat] = 100 * len(need & have) / len(need)
    total_need = len(matched) + len(missing)
    skill_score = 100 * len(matched) / total_need if total_need else 0
    text_score = similarity(resume, job)
    final = 0.6 * skill_score + 0.4 * min(text_score * 2, 100)   # weighted blend
    return final, skill_score, text_score, matched, missing, breakdown


def verdict(score):
    if score >= 75:
        return "Excellent match - apply with confidence!"
    if score >= 55:
        return "Good match - add a few missing skills to stand out."
    if score >= 35:
        return "Partial match - tailor your resume and learn the key gaps."
    return "Low match - consider building the missing skills first."


def make_chart(final, breakdown, matched, missing, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    color = "#2e9e5b" if final >= 55 else "#e0a030" if final >= 35 else "#d64545"

    # left: score donut
    axes[0].pie([final, 100 - final], colors=[color, "#e6e6e6"], startangle=90,
                counterclock=False, wedgeprops=dict(width=0.32))
    axes[0].text(0, 0.05, f"{final:.0f}%", ha="center", va="center", fontsize=30, weight="bold")
    axes[0].text(0, -0.25, "match", ha="center", va="center", fontsize=12, color="gray")
    axes[0].set_title("Overall match score")

    # right: category bars
    cats = list(breakdown)
    vals = [breakdown[c] for c in cats]
    axes[1].barh(cats, vals, color="#4c78a8")
    axes[1].set_xlim(0, 100)
    axes[1].invert_yaxis()
    for i, v in enumerate(vals):
        axes[1].text(v + 1.5, i, f"{v:.0f}%", va="center")
    axes[1].set_xlabel("% of required skills you have")
    axes[1].set_title(f"Skill coverage  ({len(matched)} matched, {len(missing)} missing)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    return fig


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Resume - Job match analyzer")
    ap.add_argument("--resume", help="path to resume .txt file")
    ap.add_argument("--job", help="path to job description .txt file")
    a = ap.parse_args()

    resume = read(a.resume) if a.resume else SAMPLE_RESUME
    job = read(a.job) if a.job else SAMPLE_JOB
    if not (a.resume and a.job):
        print("(Using built-in sample data. Use --resume and --job for your own files.)\n")

    final, skill_score, text_score, matched, missing, breakdown = analyse(resume, job)

    print("=" * 56)
    print(f"  OVERALL MATCH SCORE : {final:.1f}%")
    print(f"  Skill match         : {skill_score:.1f}%")
    print(f"  Text similarity     : {text_score:.1f}%")
    print("=" * 56)
    print(f"  {verdict(final)}\n")
    print("  Skills matched :", ", ".join(matched) or "none")
    print("  Skills missing :", ", ".join(missing) or "none")
    if missing:
        print("\n  Suggestions:")
        for s in missing[:5]:
            print(f"   - Add a project or certification that shows '{s}'.")
    print()

    out = os.path.join(HERE, "resume_match_report.png")
    fig = make_chart(final, breakdown, matched, missing, out)
    print(f"Chart saved to: {out}")
    plt.show()
