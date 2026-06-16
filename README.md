# Resume / Candidate Screening System

An NLP-based system that automatically screens resumes against job descriptions, scores each candidate, and ranks them by role fit. Built to reduce manual recruiter workload and surface the best-matched candidates faster.

-----

## What It Does

Given a pool of resumes and a job description, the system:

- Extracts skills from each resume and compares them against required and preferred skills
- Calculates a semantic similarity score between the resume text and the job description
- Combines both into a single fit score and ranks every candidate
- Flags missing required skills for each candidate
- Assigns a tier — **Strong Match**, **Potential**, or **Weak Match**

It runs independently for multiple job roles in one pass, so the same candidate pool can be screened against different positions simultaneously.

-----

## Setup

```bash
pip install -r requirements.txt
python resume_screener.py
```

Outputs are saved to the `outputs/` folder automatically.

-----

## How It Works

### 1. Skill Extraction

A taxonomy of 50+ skills across categories (languages, ML/AI, cloud, databases, visualization, soft skills) is matched against each resume using whole-word regex. This avoids false matches — for example, “r” won’t accidentally match inside other words.

### 2. Scoring — Two Signals Combined

**Skill Match Score (60% of final score)**
Checks which required and preferred skills appear in the resume. Required skills are weighted 2x heavier than preferred ones, so missing a core requirement hurts more than missing a nice-to-have.

**Semantic Similarity Score (40% of final score)**
TF-IDF vectorization is used to convert both the resume and job description into numerical representations, then cosine similarity measures how closely they align in meaning. This catches relevant experience even when exact skill keywords aren’t present.

```
Final Score = 0.60 × Skill Match + 0.40 × Semantic Similarity
```

### 3. Tier Classification

|Tier        |Score Range|
|------------|-----------|
|Strong Match|≥ 70%      |
|Potential   |45–69%     |
|Weak Match  |< 45%      |

### 4. Job Roles Supported

Three roles are configured out of the box:

- Data Scientist
- ML Engineer
- Data Analyst

Each has its own required skills list, preferred skills list, and full job description used for semantic matching.

-----

## Sample Results

**Data Scientist — Top 3**

|Rank|Candidate     |Years Exp|Fit Score|Tier     |
|----|--------------|---------|---------|---------|
|1   |Aisha Rahman  |5        |58.6%    |Potential|
|2   |James Okafor  |4        |56.1%    |Potential|
|3   |Sofia Martínez|3        |50.1%    |Potential|

**ML Engineer — Top 3**

|Rank|Candidate     |Years Exp|Fit Score|Tier      |
|----|--------------|---------|---------|----------|
|1   |Priya Nair    |6        |57.5%    |Potential |
|2   |Lucas Oliveira|4        |48.1%    |Potential |
|3   |Diana Popescu |7        |25.7%    |Weak Match|

Note: scores reflect how well a mixed candidate pool maps to each specific role — a strong Data Scientist won’t necessarily rank well for an ML Engineer position, which is the intended behaviour.

-----

## Outputs

**`screening_dashboard_<role>.png`** — One dashboard per role containing:

- Ranked candidate bar chart with tier labels and experience
- Skill match vs semantic similarity scatter plot
- Tier distribution pie chart
- Skill coverage heatmap (required vs preferred vs missing per candidate)
- Required skills match % with a pass threshold line
- Experience vs fit score scatter

**`candidates.csv`** — Full candidate pool with extracted skills.

**`ranked_<role>.csv`** — Ranked results per role with all scores broken down.

-----

## Business Value

- **Saves recruiter time** — instead of reading 100+ resumes, recruiters review a ranked shortlist
- **Consistent scoring** — every candidate is evaluated against the same criteria with no bias from reviewer fatigue
- **Skill gap visibility** — missing required skills are flagged per candidate, making interview prep easier
- **Multi-role screening** — one run screens the same pool against multiple open positions

-----

## Dependencies

```
scikit-learn
matplotlib
seaborn
pandas
numpy
```
