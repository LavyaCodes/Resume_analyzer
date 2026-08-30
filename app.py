import json
import math
import os
import re
from io import BytesIO
from typing import List, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from docx import Document
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

st.set_page_config(page_title="ResumeIQ AI", page_icon="🧠", layout="wide")


def load_css() -> None:
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_path, "r", encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)


load_css()

# ---------------------------------------------------------------------------
# ORIGINAL BACKEND LOGIC — unchanged (extraction, chunking, retrieval, scoring)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_embedding_model() -> SentenceTransformer:
    """Load the local embedding model only once."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_docx(file_bytes: bytes) -> str:
    document = Document(BytesIO(file_bytes))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def extract_text(uploaded_file) -> str:
    """Extract text from PDF, DOCX, or TXT files."""
    file_bytes = uploaded_file.getvalue()
    extension = uploaded_file.name.lower().split(".")[-1]

    if extension == "pdf":
        return clean_text(extract_pdf(file_bytes))
    if extension == "docx":
        return clean_text(extract_docx(file_bytes))
    if extension == "txt":
        return clean_text(file_bytes.decode("utf-8", errors="ignore"))

    raise ValueError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap

    return chunks


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return 0.0
    return float(np.dot(vector_a, vector_b) / denominator)


def retrieve_chunks(
    chunks: List[str], query: str, model: SentenceTransformer, top_k: int = 4
) -> Tuple[List[str], float]:
    """Embed chunks and retrieve the chunks most relevant to the job description."""
    chunk_embeddings = model.encode(chunks, normalize_embeddings=True)
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    scores = chunk_embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    retrieved = [chunks[index] for index in top_indices]
    best_score = float(scores[top_indices[0]]) if len(top_indices) else 0.0
    return retrieved, best_score


def score_to_percentage(similarity: float) -> int:
    """Convert cosine similarity into a simple classroom-friendly match score."""
    normalized = max(0.0, min(1.0, (similarity + 1) / 2))
    return round(normalized * 100)


# ---------------------------------------------------------------------------
# AI ANALYSIS — same retrieval-grounded call as before, now asked to return a
# structured JSON object (instead of freeform markdown) so the redesigned UI
# can drive real cards, bars, a radar chart, and a roadmap off real model
# output rather than inventing numbers on the frontend.
# ---------------------------------------------------------------------------


def analyze_with_gemini(job_description: str, context: str, api_key: str, model_name: str) -> dict:
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a fair and practical resume screening assistant helping a candidate
improve their resume for a specific role. Use only the retrieved resume
context given below. Do not invent qualifications that are not supported by
the context. Do not use age, gender, nationality, photo, marital status,
religion, disability, or other protected personal information in the
assessment.

TARGET ROLE / JOB DESCRIPTION:
{job_description}

RETRIEVED RESUME CONTEXT:
{context}

Respond with ONLY a single valid JSON object (no markdown fences, no
commentary, no trailing text) matching exactly this shape:

{{
  "overall_score": <integer 0-100>,
  "overall_score_label": "<Excellent|Good|Fair|Needs Work>",
  "ats_compatibility": <integer 0-100>,
  "skills_match": <integer 0-100>,
  "experience_match": <integer 0-100>,
  "skill_breakdown": [{{"category": "<string>", "score": <integer 0-100>}}],
  "radar": {{"Technical Skills": <0-100>, "Domain Knowledge": <0-100>, "Problem Solving": <0-100>, "Communication": <0-100>, "Leadership": <0-100>, "Adaptability": <0-100>}},
  "matched_skills": ["<string>"],
  "missing_skills": ["<string>"],
  "recommendations": [{{"title": "<string>", "description": "<string>", "priority": "High|Medium|Low"}}],
  "career_roadmap": [{{"period": "<string, e.g. Week 1-2>", "description": "<string>"}}],
  "projected_score": <integer 0-100>,
  "resume_strengths": ["<string>"],
  "areas_for_improvement": ["<string>"],
  "interview_questions": ["<string>"]
}}

Provide 4-6 items for "skill_breakdown", 3-5 for "recommendations" and
"career_roadmap", 3-5 for "resume_strengths" and "areas_for_improvement", and
5 for "interview_questions".
"""

    response = client.models.generate_content(model=model_name, contents=prompt)
    raw = (response.text or "{}").strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"parse_error": True, "raw_text": raw}


# ---------------------------------------------------------------------------
# UI HELPERS — small HTML-fragment builders for the redesigned components
# ---------------------------------------------------------------------------

PALETTE = ["#34d399", "#38bdf8", "#ec4899", "#fbbf24", "#8b5cf6", "#38bdf8"]
REC_ICONS = ["📘", "🧩", "🏅", "🌱", "🚀", "🔧"]


def metric_card(icon: str, value: str, label: str, sub: str, trend_up: bool = True) -> str:
    trend = '<span class="rq-metric-trend">↗</span>' if trend_up else ""
    return f"""<div class="rq-card rq-metric-card">
      <div class="rq-metric-top"><div class="rq-metric-icon">{icon}</div>{trend}</div>
      <div class="rq-metric-value">{value}</div>
      <div class="rq-metric-label">{label}</div>
      <div class="rq-metric-sub">{sub}</div>
    </div>"""


def score_card(score: int, tag: str, sub: str) -> str:
    return f"""<div class="rq-card rq-score-card">
      <div class="rq-score-ring" style="--score:{score};">
        <div class="rq-score-ring-inner"><div class="rq-score-value">{score}</div></div>
      </div>
      <div class="rq-score-label">Match Score</div>
      <div class="rq-score-tag">{tag}</div>
      <div class="rq-caption" style="text-align:center;margin-top:10px;">{sub}</div>
    </div>"""


def skill_bar(name: str, pct: int, color: str) -> str:
    pct = max(0, min(100, pct))
    return f"""<div class="rq-skillbar">
      <div class="rq-skillbar-top"><span>{name}</span><span style="color:{color};font-weight:700;">{pct}%</span></div>
      <div class="rq-skillbar-track"><div class="rq-skillbar-fill" style="width:{pct}%;background:{color};"></div></div>
    </div>"""


def pill(text: str, matched: bool) -> str:
    cls = "rq-pill-match" if matched else "rq-pill-missing"
    mark = "✓" if matched else "✗"
    return f'<span class="rq-pill {cls}">{mark} {text}</span>'


def recommendation_card(icon: str, title: str, desc: str, priority: str) -> str:
    cls = {"High": "rq-priority-high", "Medium": "rq-priority-medium", "Low": "rq-priority-low"}.get(
        priority, "rq-priority-medium"
    )
    return f"""<div class="rq-rec-card">
      <div class="rq-rec-icon">{icon}</div>
      <div class="rq-rec-body">
        <div class="rq-rec-top"><span class="rq-rec-title">{title}</span><span class="rq-priority-badge {cls}">{priority} Priority</span></div>
        <div class="rq-rec-desc">{desc}</div>
      </div>
    </div>"""


def roadmap_item(num: int, period: str, desc: str, is_last: bool) -> str:
    cls = " rq-roadmap-last" if is_last else ""
    return f"""<div class="rq-roadmap-item{cls}">
      <div class="rq-roadmap-dot">{num}</div>
      <div class="rq-roadmap-body">
        <div class="rq-roadmap-period">{period.upper()}</div>
        <div class="rq-roadmap-desc">{desc}</div>
      </div>
    </div>"""


def list_item(text: str, ok: bool) -> str:
    icon_cls = "rq-icon-ok" if ok else "rq-icon-warn"
    mark = "✓" if ok else "✗"
    return f'<div class="rq-list-item"><span class="{icon_cls}">{mark}</span><span>{text}</span></div>'


def feature_card(icon: str, color: str, title: str, desc: str) -> str:
    return f"""<div class="rq-card rq-feature-card">
      <div class="rq-feature-icon" style="background:{color}22;color:{color};">{icon}</div>
      <div class="rq-feature-title">{title}</div>
      <div class="rq-feature-desc">{desc}</div>
    </div>"""


def stepper(active_step: int) -> str:
    labels = ["Upload", "Analyze", "Results"]
    parts = []
    for i, label in enumerate(labels, start=1):
        cls = "rq-step-active" if i == active_step else ""
        parts.append(f'<div class="rq-step {cls}"><span class="rq-step-num">{i}</span> {label}</div>')
        if i < len(labels):
            parts.append('<div class="rq-step-line"></div>')
    return f'<div class="rq-stepper">{"".join(parts)}</div>'


def radar_svg(radar: dict, size: int = 260) -> str:
    labels = list(radar.keys())
    n = max(len(labels), 3)
    cx = cy = size / 2
    max_r = size / 2 - 42
    angle_step = 360 / n

    def point(index: int, radius: float):
        angle = math.radians(index * angle_step - 90)
        return cx + radius * math.cos(angle), cy + radius * math.sin(angle)

    rings = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, max_r * frac) for i in range(n)))
        rings.append(f'<polygon points="{pts}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="1" />')

    data_pts = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in (point(i, max_r * (radar[label] / 100)) for i, label in enumerate(labels))
    )
    data_poly = f'<polygon points="{data_pts}" fill="rgba(139,92,246,0.28)" stroke="#8b5cf6" stroke-width="2" />'

    label_svgs = []
    for i, label in enumerate(labels):
        x, y = point(i, max_r + 26)
        label_svgs.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" class="rq-radar-label">{label}</text>')

    return (
        f'<svg viewBox="0 0 {size} {size}" class="rq-radar-svg">'
        + "".join(rings)
        + data_poly
        + "".join(label_svgs)
        + "</svg>"
    )


def build_report_markdown(filename: str, target_role: str, data: dict) -> str:
    lines = ["# ResumeIQ AI — Analysis Report", "", f"**Resume:** {filename}", f"**Target role:** {target_role}", ""]
    lines.append(f"## Overall Score: {data.get('overall_score', '—')}/100 ({data.get('overall_score_label', '—')})")
    lines.append(f"- ATS Compatibility: {data.get('ats_compatibility', '—')}%")
    lines.append(f"- Skills Match: {data.get('skills_match', '—')}%")
    lines.append(f"- Experience Match: {data.get('experience_match', '—')}%\n")

    if data.get("matched_skills"):
        lines.append("## Matched Skills")
        lines.append(", ".join(data["matched_skills"]) + "\n")
    if data.get("missing_skills"):
        lines.append("## Skill Gaps")
        lines.append(", ".join(data["missing_skills"]) + "\n")

    if data.get("recommendations"):
        lines.append("## AI Recommendations")
        for rec in data["recommendations"]:
            lines.append(f"- **{rec.get('title', '')}** ({rec.get('priority', '')} priority): {rec.get('description', '')}")
        lines.append("")

    if data.get("career_roadmap"):
        lines.append("## Career Roadmap")
        for step in data["career_roadmap"]:
            lines.append(f"- **{step.get('period', '')}**: {step.get('description', '')}")
        lines.append("")

    if data.get("resume_strengths"):
        lines.append("## Resume Strengths")
        for item in data["resume_strengths"]:
            lines.append(f"- {item}")
        lines.append("")

    if data.get("areas_for_improvement"):
        lines.append("## Areas for Improvement")
        for item in data["areas_for_improvement"]:
            lines.append(f"- {item}")
        lines.append("")

    if data.get("interview_questions"):
        lines.append("## Practice Interview Questions")
        for i, question in enumerate(data["interview_questions"], start=1):
            lines.append(f"{i}. {question}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------

DEFAULTS = {
    "page": "home",
    "resume_filename": None,
    "resume_text": None,
    "chunks": [],
    "retrieved_chunks": [],
    "match_score": None,
    "analysis": None,
    "analysis_error": None,
    "target_role_value": "",
}
for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def go(page: str) -> None:
    st.session_state.page = page
    st.rerun()


# ---------------------------------------------------------------------------
# NAVBAR & SIDEBAR
# ---------------------------------------------------------------------------


def render_navbar() -> None:
    with st.container(key="rq_navbar"):
        cols = st.columns([2.4, 0.8, 1.1, 1, 0.9, 1.2])
        with cols[0]:
            st.markdown(
                '<div class="rq-logo">🧠 ResumeIQ <span class="rq-ai">AI</span></div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            if st.button("🏠 Home", key="nav_home", use_container_width=True):
                go("home")
        with cols[2]:
            if st.button("ℹ️ How It Works", key="nav_how", use_container_width=True):
                go("home")
        with cols[3]:
            if st.button("📊 Dashboard", key="nav_dash", use_container_width=True):
                go("results" if st.session_state.analysis is not None or st.session_state.resume_filename else "upload")
        with cols[4]:
            if st.button("👤 Profile", key="nav_profile", use_container_width=True):
                go("profile")
        with cols[5]:
            if st.button("Analyze Resume", key="nav_cta", use_container_width=True):
                go("upload")


def render_sidebar() -> None:
    with st.sidebar:
        st.header("AI Settings")
        st.text_input(
            "Gemini API key",
            value=os.getenv("GEMINI_API_KEY", ""),
            type="password",
            help="You may also store it in a .env file.",
            key="api_key",
        )
        st.text_input("Gemini model", value="gemini-3.5-flash", key="model_name")
        st.slider("Retrieved chunks", min_value=2, max_value=8, value=4, key="top_k")
        st.info("The embedding model runs locally. Gemini generates the structured breakdown shown in the results.")


# ---------------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------------


def render_home() -> None:
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        st.markdown('<div class="rq-badge">✦ AI-POWERED CAREER INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="rq-hero-title">Your Resume.<br>Your Career.<br>'
            '<span class="rq-grad">Powered by AI.</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="rq-hero-sub">ResumeIQ deeply analyzes your resume, identifies critical skill gaps, '
            "and delivers personalized career recommendations — so you land your dream role faster.</div>",
            unsafe_allow_html=True,
        )
        btn_col1, btn_col2 = st.columns([1.3, 1])
        with btn_col1:
            if st.button("⚡ Analyze My Resume →", key="cta_home", use_container_width=True):
                go("upload")
        with btn_col2:
            st.markdown(
                '<a href="#features" class="rq-btn-secondary">See How It Works ›</a>',
                unsafe_allow_html=True,
            )
        st.markdown(
            """<div class="rq-stats-row">
                 <div><div class="rq-stat-num">50K+</div><div class="rq-stat-label">Resumes Analyzed</div></div>
                 <div><div class="rq-stat-num">94%</div><div class="rq-stat-label">User Satisfaction</div></div>
                 <div><div class="rq-stat-num">3×</div><div class="rq-stat-label">More Interviews</div></div>
               </div>""",
            unsafe_allow_html=True,
        )

    with right:
        mock_bars = skill_bar("ATS Score", 92, "#34d399") + skill_bar("Skills Match", 76, "#ec4899") + skill_bar("Experience", 68, "#38bdf8")
        mock_pills = "".join(pill(s, True) for s in ["Python", "TensorFlow", "PyTorch", "SQL", "Docker"]) + pill("Kubernetes", False)
        st.markdown(
            f"""<div class="rq-hero-mock">
                 <div class="rq-hero-mock-file">
                   <div class="rq-hero-mock-file-icon">📄</div>
                   <div>
                     <div class="rq-hero-mock-filename">ML_Engineer_Resume.pdf</div>
                     <div class="rq-hero-mock-filesub">Analyzed just now</div>
                   </div>
                 </div>
                 <div class="rq-hero-mock-body">
                   <div class="rq-score-ring rq-score-ring-sm" style="--score:79;">
                     <div class="rq-score-ring-inner rq-score-ring-inner-sm"><div class="rq-score-value rq-score-value-sm">79</div></div>
                   </div>
                   <div class="rq-hero-mock-bars">{mock_bars}</div>
                 </div>
                 <div class="rq-pill-row">{mock_pills}</div>
                 <div class="rq-hero-mock-quote">✨ "Strong ML profile. Adding MLOps expertise could boost your match score to <strong>94%</strong>."</div>
                 <div class="rq-floating-badge rq-floating-badge-1">✔ 10 Skills Matched</div>
                 <div class="rq-floating-badge rq-floating-badge-2">🛡 ATS Safe · 92%</div>
                 <div class="rq-floating-badge rq-floating-badge-3">📈 +23% Potential</div>
               </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div id="features"></div>', unsafe_allow_html=True)
    st.markdown(
        """<div style="text-align:center;margin-top:70px;">
             <div class="rq-badge">EVERYTHING YOU NEED</div>
             <h2 style="font-size:2.2rem;margin:6px 0 10px 0;">Built for the modern job seeker</h2>
             <p class="rq-caption" style="font-size:1rem;">Comprehensive AI tools for every stage of your job search journey.</p>
           </div>""",
        unsafe_allow_html=True,
    )
    feature_cols = st.columns(4, gap="medium")
    features = [
        ("🧠", "#8b5cf6", "AI Resume Analysis", "Deep semantic analysis against your target role using state-of-the-art language models."),
        ("🎯", "#ec4899", "Skill Gap Detection", "Identify exactly what skills you are missing with a precise plan to fill them fast."),
        ("🛡️", "#38bdf8", "ATS Optimization", "Ensure your resume passes automated screening systems used by top-tier companies."),
        ("📈", "#34d399", "Career Roadmap", "Get a personalized week-by-week action plan to significantly accelerate your career."),
    ]
    for col, (icon, color, title, desc) in zip(feature_cols, features):
        with col:
            st.markdown(feature_card(icon, color, title, desc), unsafe_allow_html=True)


def render_upload() -> None:
    st.markdown(stepper(1), unsafe_allow_html=True)
    st.markdown(
        '<h1 style="text-align:center;">Upload Your Resume</h1>'
        '<p class="rq-caption" style="text-align:center;font-size:1rem;">'
        "Upload your resume and target role for a deep AI-powered analysis.</p>",
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([0.5, 3, 0.5])
    with mid:
        with st.container(key="upload_dropzone"):
            st.markdown(
                """<div class="rq-dropzone-head">
                     <div class="rq-dropzone-icon">⬆️</div>
                     <div class="rq-dropzone-title">Drop your resume here</div>
                     <div class="rq-caption">or click below to browse your files</div>
                     <div class="rq-pill-row" style="justify-content:center;">
                       <span class="rq-pill rq-pill-file">📄 PDF</span>
                       <span class="rq-pill rq-pill-file">📄 DOCX</span>
                       <span class="rq-pill rq-pill-file">📄 TXT</span>
                     </div>
                   </div>""",
                unsafe_allow_html=True,
            )
            resume_file = st.file_uploader(
                "Resume file", type=["pdf", "docx", "txt"], label_visibility="collapsed", key="resume_upload"
            )

        st.markdown('<div class="rq-section-title" style="margin-top:26px;">Target Job Role</div>', unsafe_allow_html=True)
        target_role = st.text_input(
            "Target Job Role", placeholder="e.g. Machine Learning Engineer", label_visibility="collapsed", key="target_role"
        )

        st.markdown(
            '<div class="rq-section-title" style="margin-top:16px;">Job Description '
            '<span class="rq-caption" style="display:inline;">(Optional — improves accuracy)</span></div>',
            unsafe_allow_html=True,
        )
        job_description = st.text_area(
            "Job Description",
            height=180,
            placeholder="Paste the job description here to get a more accurate analysis...",
            label_visibility="collapsed",
            key="job_description",
        )

        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("⚡ Analyze Resume", key="cta_analyze", use_container_width=True)

    if not analyze_clicked:
        return

    if resume_file is None:
        st.error("Please upload a resume.")
        return
    if not target_role.strip():
        st.error("Please enter a target job role.")
        return

    combined_query = (target_role + "\n\n" + job_description).strip()

    try:
        with st.spinner("Reading the resume..."):
            resume_text = extract_text(resume_file)

        if len(resume_text.split()) < 20:
            st.error("Very little text was extracted. The PDF may be scanned. Try a DOCX or text-based PDF.")
            return

        chunks = chunk_text(resume_text)
        embedding_model = load_embedding_model()

        with st.spinner("Finding the most relevant resume sections..."):
            retrieved_chunks, best_similarity = retrieve_chunks(
                chunks, combined_query, embedding_model, top_k=st.session_state.top_k
            )

        match_score = score_to_percentage(best_similarity)
        context = "\n\n--- RETRIEVED SECTION ---\n".join(retrieved_chunks)

        analysis = None
        analysis_error = None
        if st.session_state.api_key:
            with st.spinner("Generating grounded feedback..."):
                analysis = analyze_with_gemini(
                    job_description=combined_query,
                    context=context,
                    api_key=st.session_state.api_key,
                    model_name=st.session_state.model_name,
                )
            if analysis.get("parse_error"):
                analysis_error = "The AI response couldn't be parsed as structured data. Showing retrieval results only."
                analysis = None
        else:
            analysis_error = "Add a Gemini API key in the sidebar to generate the full AI-powered breakdown."

        st.session_state.resume_filename = resume_file.name
        st.session_state.resume_text = resume_text
        st.session_state.chunks = chunks
        st.session_state.retrieved_chunks = retrieved_chunks
        st.session_state.match_score = match_score
        st.session_state.analysis = analysis
        st.session_state.analysis_error = analysis_error
        st.session_state.target_role_value = target_role
        go("results")

    except Exception as error:
        st.error(f"Analysis failed: {error}")
        st.exception(error)


def render_results() -> None:
    if st.session_state.resume_filename is None:
        st.info("No analysis yet — upload a resume to get started.")
        if st.button("Go to Upload", key="cta_secondary"):
            go("upload")
        return

    data = st.session_state.analysis or {}
    match_score = st.session_state.match_score
    overall_score = data.get("overall_score", match_score)
    role = st.session_state.get("target_role_value", "")

    header_left, header_right = st.columns([3, 1.6])
    with header_left:
        st.markdown('<div class="rq-caption" style="margin-bottom:0;">Analysis Report</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:1.4rem;font-weight:800;font-family:Poppins,sans-serif;">'
            f'{st.session_state.resume_filename} <span class="rq-caption" style="font-weight:500;">→ {role or "Target role"}</span></div>',
            unsafe_allow_html=True,
        )
    with header_right:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("↑ Analyze Another", key="cta_secondary", use_container_width=True):
                for k, v in DEFAULTS.items():
                    if k != "page":
                        st.session_state[k] = v
                go("upload")
        with b2:
            report = build_report_markdown(st.session_state.resume_filename, role, data)
            st.download_button(
                "⬇ Download Report",
                data=report,
                file_name="resumeiq_report.md",
                mime="text/markdown",
                key="cta_download",
                use_container_width=True,
            )

    if st.session_state.analysis_error:
        st.warning(st.session_state.analysis_error)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    score_col, metrics_col = st.columns([1, 2], gap="medium")
    with score_col:
        st.markdown(
            score_card(
                overall_score,
                data.get("overall_score_label", "Good" if overall_score and overall_score >= 60 else "Needs Work"),
                f"Based on {role or 'your target'} role requirements",
            ),
            unsafe_allow_html=True,
        )
    with metrics_col:
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                metric_card("⭐", f"{overall_score}/100", "Overall Score", "AI-generated holistic score"),
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                metric_card("🛡️", f"{data.get('ats_compatibility', '—')}%", "ATS Compatibility", "Passes top ATS systems"),
                unsafe_allow_html=True,
            )
        m3, m4 = st.columns(2)
        with m3:
            st.markdown(
                metric_card("🎯", f"{data.get('skills_match', match_score)}%", "Skills Match", "Semantic + skill overlap", trend_up=False),
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                metric_card("⏱️", f"{data.get('experience_match', '—')}%", "Experience Match", "Estimated vs. role requirement"),
                unsafe_allow_html=True,
            )

    if data:
        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        bd_col, radar_col = st.columns([1.3, 1], gap="medium")
        with bd_col:
            bars_html = "".join(
                skill_bar(item.get("category", ""), item.get("score", 0), PALETTE[i % len(PALETTE)])
                for i, item in enumerate(data.get("skill_breakdown", []))
            )
            st.markdown(f'<div class="rq-card"><div class="rq-section-title">Skill Breakdown</div>{bars_html}</div>', unsafe_allow_html=True)
        with radar_col:
            radar_html = radar_svg(data["radar"]) if data.get("radar") else ""
            st.markdown(
                f'<div class="rq-card" style="text-align:center;"><div class="rq-section-title">Skill Profile</div>{radar_html}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        match_col, gap_col = st.columns(2, gap="medium")
        with match_col:
            matched_html = "".join(pill(s, True) for s in data.get("matched_skills", []))
            st.markdown(
                f"""<div class="rq-card">
                      <div class="rq-section-title">✓ Matched Skills <span class="rq-pill rq-pill-match" style="margin-left:8px;">{len(data.get('matched_skills', []))} found</span></div>
                      <div class="rq-pill-row">{matched_html}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
        with gap_col:
            missing_html = "".join(pill(s, False) for s in data.get("missing_skills", []))
            st.markdown(
                f"""<div class="rq-card">
                      <div class="rq-section-title">⚠ Skill Gaps <span class="rq-pill rq-pill-missing" style="margin-left:8px;">{len(data.get('missing_skills', []))} missing</span></div>
                      <div class="rq-pill-row">{missing_html}</div>
                      <div class="rq-caption" style="margin-top:10px;">These skills appear frequently for this role and are absent from your resume.</div>
                    </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        rec_col, road_col = st.columns(2, gap="medium")
        with rec_col:
            recs_html = "".join(
                recommendation_card(REC_ICONS[i % len(REC_ICONS)], rec.get("title", ""), rec.get("description", ""), rec.get("priority", "Medium"))
                for i, rec in enumerate(data.get("recommendations", []))
            )
            st.markdown(f'<div class="rq-card"><div class="rq-section-title">✨ AI Recommendations</div>{recs_html}</div>', unsafe_allow_html=True)
        with road_col:
            roadmap = data.get("career_roadmap", [])
            roadmap_html = "".join(
                roadmap_item(i, step.get("period", ""), step.get("description", ""), i == len(roadmap))
                for i, step in enumerate(roadmap, start=1)
            )
            projected_html = ""
            if data.get("projected_score") is not None:
                delta = max(0, data["projected_score"] - (overall_score or 0))
                projected_html = f"""<div class="rq-roadmap-projected">
                      <div><div style="font-weight:700;">Projected Match Score</div><div class="rq-caption">After completing this roadmap</div></div>
                      <div style="text-align:right;">
                        <div style="color:var(--rq-green);font-size:1.7rem;font-weight:800;">{data['projected_score']}</div>
                        <div class="rq-caption">+{delta} points</div>
                      </div>
                    </div>"""
            st.markdown(
                f'<div class="rq-card"><div class="rq-section-title">📈 Career Roadmap</div>{roadmap_html}{projected_html}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
        strength_col, improve_col = st.columns(2, gap="medium")
        with strength_col:
            strengths_html = "".join(list_item(item, True) for item in data.get("resume_strengths", []))
            st.markdown(f'<div class="rq-card"><div class="rq-section-title">⭐ Resume Strengths</div>{strengths_html}</div>', unsafe_allow_html=True)
        with improve_col:
            improve_html = "".join(list_item(item, False) for item in data.get("areas_for_improvement", []))
            st.markdown(f'<div class="rq-card"><div class="rq-section-title">⚠ Areas for Improvement</div>{improve_html}</div>', unsafe_allow_html=True)

        if data.get("interview_questions"):
            with st.expander("🎤 Practice Interview Questions"):
                for i, question in enumerate(data["interview_questions"], start=1):
                    st.markdown(f"**{i}.** {question}")

    with st.expander("View retrieved resume evidence"):
        for index, chunk in enumerate(st.session_state.retrieved_chunks, start=1):
            st.markdown(f"**Evidence {index}**")
            st.write(chunk)
            st.divider()

    with st.expander("View extracted resume text"):
        st.write(st.session_state.resume_text)


def render_profile() -> None:
    st.markdown('<h1 style="text-align:center;">Profile</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rq-card" style="max-width:520px;margin:0 auto;text-align:center;">'
        '<div class="rq-caption">Profile settings are coming soon. This is a placeholder page.</div>'
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------


def main() -> None:
    render_sidebar()
    render_navbar()

    page = st.session_state.page
    if page == "home":
        render_home()
    elif page == "upload":
        render_upload()
    elif page == "results":
        render_results()
    elif page == "profile":
        render_profile()
    else:
        render_home()


if __name__ == "__main__":
    main()
