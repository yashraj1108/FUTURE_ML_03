import re
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)


# ── 1. SKILL TAXONOMY ─────────────────────────────────────────────────────────

ALL_SKILLS = {
    'languages':      ['python', 'java', 'javascript', 'typescript', 'c++', 'r', 'sql', 'scala', 'go', 'rust'],
    'ml_ai':          ['machine learning', 'deep learning', 'nlp', 'computer vision', 'tensorflow',
                       'pytorch', 'keras', 'scikit-learn', 'xgboost', 'transformers', 'llm'],
    'data':           ['pandas', 'numpy', 'spark', 'hadoop', 'dbt', 'airflow', 'kafka', 'etl',
                       'data pipeline', 'feature engineering'],
    'cloud':          ['aws', 'gcp', 'azure', 'docker', 'kubernetes', 'terraform', 'ci/cd', 'mlops'],
    'databases':      ['postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'bigquery', 'snowflake'],
    'visualization':  ['tableau', 'power bi', 'matplotlib', 'plotly', 'd3.js', 'looker'],
    'soft_skills':    ['communication', 'leadership', 'teamwork', 'problem solving', 'agile', 'scrum'],
}

FLAT_SKILLS = [skill for group in ALL_SKILLS.values() for skill in group]


# ── 2. JOB DESCRIPTIONS ───────────────────────────────────────────────────────

JOB_DESCRIPTIONS = {
    'Data Scientist': {
        'description': """
        We are looking for a Data Scientist to join our analytics team.
        You will build machine learning models, analyse large datasets, and
        deliver actionable insights to drive product decisions.

        Requirements:
        - Strong proficiency in Python and SQL
        - Experience with machine learning frameworks: scikit-learn, TensorFlow or PyTorch
        - Hands-on experience with NLP or computer vision projects
        - Familiarity with feature engineering and model evaluation
        - Knowledge of pandas, numpy for data manipulation
        - Experience with cloud platforms (AWS, GCP, or Azure)
        - Strong communication skills to present findings to non-technical stakeholders
        - Experience with data visualization tools (Tableau, Power BI, or Plotly)
        - Bonus: MLOps experience, Spark, or Kafka
        """,
        'required':  ['python', 'sql', 'machine learning', 'scikit-learn', 'pandas', 'numpy',
                      'feature engineering', 'communication'],
        'preferred': ['tensorflow', 'pytorch', 'nlp', 'aws', 'gcp', 'tableau', 'plotly',
                      'deep learning', 'spark'],
        'weight_required': 2.0,
        'weight_preferred': 1.0,
    },
    'ML Engineer': {
        'description': """
        We are hiring a Machine Learning Engineer to design and deploy scalable ML systems.
        You will own the full ML lifecycle from data ingestion to model serving in production.

        Requirements:
        - Proficiency in Python and at least one of: Scala, Go, or Java
        - Deep understanding of ML frameworks: PyTorch, TensorFlow, or Keras
        - Strong experience with MLOps: Docker, Kubernetes, CI/CD pipelines
        - Cloud deployment experience on AWS, GCP, or Azure
        - Experience with data pipelines: Airflow, Kafka, or Spark
        - Familiarity with SQL and NoSQL databases
        - Understanding of transformers and LLMs is a strong plus
        - Agile development practices
        """,
        'required':  ['python', 'machine learning', 'docker', 'kubernetes', 'aws', 'sql',
                      'airflow', 'mlops', 'agile'],
        'preferred': ['pytorch', 'tensorflow', 'kafka', 'spark', 'transformers', 'llm',
                      'gcp', 'scala', 'ci/cd'],
        'weight_required': 2.0,
        'weight_preferred': 1.0,
    },
    'Data Analyst': {
        'description': """
        We are looking for a Data Analyst who can turn raw data into clear business insights.
        You will work closely with product, marketing, and operations teams.

        Requirements:
        - Strong SQL skills for data extraction and analysis
        - Proficiency in Python or R for statistical analysis
        - Experience with BI tools: Tableau, Power BI, or Looker
        - Good understanding of data warehousing concepts (Snowflake, BigQuery)
        - Excellent communication and presentation skills
        - Experience building dashboards and reports
        - Problem solving mindset with attention to detail
        """,
        'required':  ['sql', 'python', 'tableau', 'communication', 'problem solving'],
        'preferred': ['r', 'power bi', 'looker', 'snowflake', 'bigquery', 'plotly', 'agile'],
        'weight_required': 2.0,
        'weight_preferred': 1.0,
    },
}


# ── 3. SYNTHETIC RESUME GENERATOR ─────────────────────────────────────────────

CANDIDATE_PROFILES = [
    # strong DS candidates
    {'name': 'Aisha Rahman',    'years': 5, 'role': 'Data Scientist',
     'skills': ['python', 'machine learning', 'deep learning', 'nlp', 'tensorflow', 'scikit-learn',
                'pandas', 'numpy', 'sql', 'aws', 'tableau', 'feature engineering', 'communication']},
    {'name': 'James Okafor',    'years': 4, 'role': 'Data Scientist',
     'skills': ['python', 'machine learning', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
                'sql', 'gcp', 'plotly', 'feature engineering', 'communication', 'nlp']},
    {'name': 'Sofia Martínez',  'years': 3, 'role': 'Data Scientist',
     'skills': ['python', 'scikit-learn', 'pandas', 'numpy', 'sql', 'machine learning',
                'feature engineering', 'tableau', 'communication']},
    # mid-level
    {'name': 'Chen Wei',        'years': 2, 'role': 'Junior Data Scientist',
     'skills': ['python', 'pandas', 'numpy', 'sql', 'scikit-learn', 'machine learning',
                'matplotlib', 'communication']},
    {'name': 'Priya Nair',      'years': 6, 'role': 'ML Engineer',
     'skills': ['python', 'tensorflow', 'pytorch', 'docker', 'kubernetes', 'aws', 'mlops',
                'airflow', 'kafka', 'sql', 'machine learning', 'ci/cd', 'agile', 'transformers']},
    {'name': 'Lucas Oliveira',  'years': 4, 'role': 'ML Engineer',
     'skills': ['python', 'pytorch', 'docker', 'kubernetes', 'gcp', 'mlops', 'airflow',
                'sql', 'machine learning', 'agile', 'ci/cd']},
    {'name': 'Fatima Al-Zahra', 'years': 3, 'role': 'Data Analyst',
     'skills': ['sql', 'python', 'tableau', 'power bi', 'communication', 'problem solving',
                'snowflake', 'bigquery', 'agile']},
    {'name': 'Marcus Johnson',  'years': 5, 'role': 'Data Analyst',
     'skills': ['sql', 'tableau', 'power bi', 'looker', 'python', 'communication',
                'problem solving', 'bigquery', 'plotly']},
    # weaker / mismatched candidates
    {'name': 'Lena Müller',     'years': 1, 'role': 'Student',
     'skills': ['python', 'sql', 'pandas', 'numpy', 'communication']},
    {'name': 'Raj Patel',       'years': 2, 'role': 'Software Engineer',
     'skills': ['javascript', 'typescript', 'java', 'sql', 'docker', 'aws', 'agile', 'communication']},
    {'name': 'Yuki Tanaka',     'years': 3, 'role': 'Backend Developer',
     'skills': ['python', 'java', 'go', 'postgresql', 'redis', 'docker', 'kubernetes', 'agile']},
    {'name': 'Diana Popescu',   'years': 7, 'role': 'Senior Data Engineer',
     'skills': ['python', 'sql', 'spark', 'kafka', 'airflow', 'aws', 'dbt', 'etl',
                'data pipeline', 'snowflake', 'postgresql']},
]

EDUCATION_OPTIONS = [
    "BSc Computer Science, University of Lagos (2019)",
    "MSc Data Science, Technical University of Munich (2021)",
    "BSc Statistics, University of Cape Town (2020)",
    "MSc Machine Learning, Imperial College London (2022)",
    "BSc Mathematics, University of São Paulo (2018)",
    "BSc Software Engineering, IIT Delhi (2020)",
    "MSc Artificial Intelligence, University of Amsterdam (2021)",
    "BSc Information Systems, University of Warsaw (2019)",
]

EXPERIENCE_TEMPLATES = [
    "Developed {skill1} and {skill2} pipelines that reduced processing time by {pct}%",
    "Built and deployed {skill1} models achieving {pct}% improvement in prediction accuracy",
    "Led a team of {n} engineers to deliver {skill1} and {skill2} solutions",
    "Designed scalable {skill1} infrastructure handling {n}M+ records daily",
    "Collaborated with cross-functional teams to implement {skill1} driven insights",
    "Maintained and optimised {skill1} workflows improving efficiency by {pct}%",
    "Created {skill1} dashboards and reports used by {n}0+ stakeholders",
    "Researched and prototyped {skill1} approaches for {skill2} use cases",
]


def build_resume_text(profile):
    name = profile['name']
    years = profile['years']
    skills = profile['skills']
    edu = random.choice(EDUCATION_OPTIONS)

    lines = [
        f"Name: {name}",
        f"Experience: {years} years",
        f"Education: {edu}",
        "",
        "Skills: " + ", ".join(skills),
        "",
        "Professional Experience:",
    ]

    n_jobs = min(years, 3)
    for _ in range(n_jobs):
        skill_sample = random.sample(skills, min(2, len(skills)))
        template = random.choice(EXPERIENCE_TEMPLATES)
        line = template.format(
            skill1=skill_sample[0],
            skill2=skill_sample[-1],
            pct=random.randint(15, 55),
            n=random.randint(2, 8),
        )
        lines.append(f"- {line}")

    lines += ["", "Certifications & Projects:"]
    if 'aws' in skills:      lines.append("- AWS Certified Machine Learning Specialty")
    if 'gcp' in skills:      lines.append("- Google Professional Data Engineer")
    if 'tableau' in skills:  lines.append("- Tableau Desktop Certified Associate")
    if 'machine learning' in skills:
        lines.append("- Kaggle Competition: Top 10% in tabular ML challenge")
    if len(lines[-3:]) == 2: lines.append("- Internal project: automated reporting pipeline")

    return "\n".join(lines)


def build_dataset():
    rows = []
    for profile in CANDIDATE_PROFILES:
        text = build_resume_text(profile)
        rows.append({
            'candidate_id': f'C{100 + len(rows):03d}',
            'name':         profile['name'],
            'years_exp':    profile['years'],
            'current_role': profile['role'],
            'skills':       profile['skills'],
            'resume_text':  text,
        })
    return pd.DataFrame(rows)


# ── 4. TEXT CLEANING ──────────────────────────────────────────────────────────

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s/]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── 5. SKILL EXTRACTION ───────────────────────────────────────────────────────

def extract_skills(text):
    text_lower = text.lower()
    found = []
    for skill in FLAT_SKILLS:
        # whole-word match so "r" doesn't match inside other words
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


# ── 6. SCORING ENGINE ─────────────────────────────────────────────────────────

def score_candidate(resume_text, resume_skills, job_key):
    job = JOB_DESCRIPTIONS[job_key]
    required  = job['required']
    preferred = job['preferred']
    w_req     = job['weight_required']
    w_pref    = job['weight_preferred']

    # skill match scores
    req_matched   = [s for s in required  if s in resume_skills]
    pref_matched  = [s for s in preferred if s in resume_skills]
    req_missing   = [s for s in required  if s not in resume_skills]
    pref_missing  = [s for s in preferred if s not in resume_skills]

    req_score  = len(req_matched)  / len(required)  if required  else 0
    pref_score = len(pref_matched) / len(preferred) if preferred else 0

    skill_score = (req_score * w_req + pref_score * w_pref) / (w_req + w_pref)

    # TF-IDF cosine similarity between resume and job description
    jd_text = clean_text(job['description'])
    res_text = clean_text(resume_text)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([jd_text, res_text])
        cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except Exception:
        cos_sim = 0.0

    # combined score (60% skill match, 40% semantic similarity)
    final_score = 0.60 * skill_score + 0.40 * cos_sim

    return {
        'skill_score':    round(skill_score * 100, 1),
        'semantic_score': round(cos_sim * 100, 1),
        'final_score':    round(final_score * 100, 1),
        'req_matched':    req_matched,
        'pref_matched':   pref_matched,
        'req_missing':    req_missing,
        'pref_missing':   pref_missing,
        'match_pct_req':  round(len(req_matched) / len(required) * 100, 1) if required else 0,
    }


def rank_candidates(df, job_key):
    results = []
    for _, row in df.iterrows():
        scores = score_candidate(row['resume_text'], row['skills'], job_key)
        results.append({
            'candidate_id':   row['candidate_id'],
            'name':           row['name'],
            'years_exp':      row['years_exp'],
            'current_role':   row['current_role'],
            **scores,
        })

    ranked = pd.DataFrame(results).sort_values('final_score', ascending=False).reset_index(drop=True)
    ranked['rank'] = ranked.index + 1

    # tier labels
    def tier(score):
        if score >= 70: return 'Strong Match'
        if score >= 45: return 'Potential'
        return 'Weak Match'

    ranked['tier'] = ranked['final_score'].apply(tier)
    return ranked


# ── 7. VISUALIZATIONS ─────────────────────────────────────────────────────────

COLORS = {
    'blue':   '#1B6CA8',
    'orange': '#F4A261',
    'green':  '#2E9E6B',
    'red':    '#E63946',
    'purple': '#7B2D8B',
    'bg':     '#F8F9FA',
    'grid':   '#E0E0E0',
    'dark':   '#1A1A2E',
    'mid':    '#555570',
}

TIER_COLORS = {'Strong Match': COLORS['green'], 'Potential': COLORS['orange'], 'Weak Match': COLORS['red']}


def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(COLORS['bg'])
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['mid'], labelsize=9)
    ax.grid(axis='x', color=COLORS['grid'], linewidth=0.6, linestyle='--')
    if title:  ax.set_title(title,  fontsize=11, fontweight='bold', color=COLORS['dark'], pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9,  color=COLORS['mid'])
    if ylabel: ax.set_ylabel(ylabel, fontsize=9,  color=COLORS['mid'])


def plot_dashboard(ranked, job_key):
    fig = plt.figure(figsize=(20, 26), facecolor='white')
    fig.suptitle(f'Resume Screening Dashboard — {job_key}',
                 fontsize=18, fontweight='bold', color=COLORS['dark'], y=0.98)

    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35,
                           top=0.94, bottom=0.04, left=0.06, right=0.97)

    top10 = ranked.head(10)

    # --- candidate ranking bar chart ---
    ax1 = fig.add_subplot(gs[0, :])
    bar_colors = [TIER_COLORS[t] for t in top10['tier']]
    bars = ax1.barh(top10['name'][::-1], top10['final_score'][::-1],
                    color=bar_colors[::-1], edgecolor='white', linewidth=0.7, zorder=3, height=0.65)

    for bar, row in zip(bars, top10.iloc[::-1].itertuples()):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{row.final_score:.1f}%  |  {row.tier}  |  {row.years_exp}yr exp',
                 va='center', fontsize=8.5, color=COLORS['dark'])

    ax1.set_xlim(0, 115)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    style_ax(ax1, f'Candidate Ranking — Overall Fit Score (Top 10)', xlabel='Fit Score (%)')
    ax1.grid(axis='x', color=COLORS['grid'], linewidth=0.6, linestyle='--')

    # legend
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=COLORS['green'],  label='Strong Match (≥70%)'),
                         Patch(color=COLORS['orange'], label='Potential (45–69%)'),
                         Patch(color=COLORS['red'],    label='Weak Match (<45%)')],
               fontsize=9, loc='lower right', framealpha=0.9)

    # --- skill score vs semantic score scatter ---
    ax2 = fig.add_subplot(gs[1, 0])
    for tier, group in ranked.groupby('tier'):
        ax2.scatter(group['skill_score'], group['semantic_score'],
                    color=TIER_COLORS[tier], s=90, label=tier, zorder=4, edgecolors='white', linewidth=0.8)
        for _, row in group.iterrows():
            ax2.annotate(row['name'].split()[0], (row['skill_score'], row['semantic_score']),
                         fontsize=7, color=COLORS['mid'],
                         xytext=(4, 3), textcoords='offset points')

    ax2.set_facecolor(COLORS['bg'])
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.spines[['left', 'bottom']].set_color(COLORS['grid'])
    ax2.tick_params(colors=COLORS['mid'], labelsize=9)
    ax2.set_title('Skill Match vs Semantic Similarity', fontsize=11, fontweight='bold',
                  color=COLORS['dark'], pad=10)
    ax2.set_xlabel('Skill Match Score (%)', fontsize=9, color=COLORS['mid'])
    ax2.set_ylabel('Semantic Similarity (%)', fontsize=9, color=COLORS['mid'])
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax2.grid(color=COLORS['grid'], linewidth=0.5, linestyle='--')
    ax2.legend(fontsize=8, framealpha=0.9)

    # --- tier distribution pie ---
    ax3 = fig.add_subplot(gs[1, 1])
    tier_counts = ranked['tier'].value_counts()
    wedge_colors = [TIER_COLORS.get(t, COLORS['blue']) for t in tier_counts.index]
    wedges, texts, autotexts = ax3.pie(
        tier_counts.values, labels=tier_counts.index,
        colors=wedge_colors, autopct='%1.0f%%',
        startangle=140, pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    for t in texts:      t.set_fontsize(10); t.set_color(COLORS['dark'])
    for a in autotexts:  a.set_fontsize(9);  a.set_color('white'); a.set_fontweight('bold')
    ax3.set_facecolor(COLORS['bg'])
    ax3.set_title('Candidate Tier Distribution', fontsize=11, fontweight='bold',
                  color=COLORS['dark'], pad=10)

    # --- required skill coverage heatmap (top 8 candidates) ---
    ax4 = fig.add_subplot(gs[2, :])
    job = JOB_DESCRIPTIONS[job_key]
    all_skills_jd = job['required'] + job['preferred']
    top8 = ranked.head(8)

    # get the actual skills list from the dataset
    heatmap_data = []
    for _, row in top8.iterrows():
        matched_req  = set(row['req_matched'])
        matched_pref = set(row['pref_matched'])
        skill_row = []
        for skill in all_skills_jd:
            if skill in matched_req:   skill_row.append(2)   # required & matched
            elif skill in matched_pref: skill_row.append(1)  # preferred & matched
            else:                       skill_row.append(0)  # missing
        heatmap_data.append(skill_row)

    hm_df = pd.DataFrame(heatmap_data, index=top8['name'], columns=all_skills_jd)

    cmap = matplotlib.colors.ListedColormap(['#FFCCCC', '#FFF3CD', '#C8E6C9'])
    sns.heatmap(hm_df, ax=ax4, cmap=cmap, linewidths=0.5, linecolor='white',
                cbar=False, annot=False)

    # custom legend
    from matplotlib.patches import Patch
    ax4.legend(handles=[
        Patch(color='#C8E6C9', label='Required — Matched'),
        Patch(color='#FFF3CD', label='Preferred — Matched'),
        Patch(color='#FFCCCC', label='Missing'),
    ], fontsize=8.5, loc='upper right', bbox_to_anchor=(1.01, 1.18), framealpha=0.9)

    ax4.set_title('Skill Coverage Map — Top 8 Candidates', fontsize=11,
                  fontweight='bold', color=COLORS['dark'], pad=10)
    ax4.set_xlabel('Job Skills', fontsize=9, color=COLORS['mid'])
    ax4.tick_params(axis='x', labelsize=7.5, rotation=35)
    ax4.tick_params(axis='y', labelsize=8.5)

    # mark required vs preferred on x-axis
    for i, skill in enumerate(all_skills_jd):
        color = COLORS['red'] if skill in job['required'] else COLORS['orange']
        ax4.get_xticklabels()[i].set_color(color)

    # --- required skill match % for top candidates ---
    ax5 = fig.add_subplot(gs[3, 0])
    top6 = ranked.head(6)
    bars5 = ax5.barh(top6['name'][::-1], top6['match_pct_req'][::-1],
                     color=[TIER_COLORS[t] for t in top6['tier'][::-1]],
                     edgecolor='white', linewidth=0.7, zorder=3, height=0.6)
    for bar, val in zip(bars5, top6['match_pct_req'][::-1]):
        ax5.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f'{val:.0f}%', va='center', fontsize=9, fontweight='bold', color=COLORS['dark'])
    ax5.set_xlim(0, 115)
    ax5.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax5.axvline(70, color=COLORS['red'], linewidth=1.5, linestyle='--', alpha=0.7)
    ax5.text(71, 0.1, 'Threshold', fontsize=7.5, color=COLORS['red'])
    style_ax(ax5, 'Required Skills Match %', xlabel='% of Required Skills Matched')

    # --- experience vs score ---
    ax6 = fig.add_subplot(gs[3, 1])
    for tier, group in ranked.groupby('tier'):
        ax6.scatter(group['years_exp'], group['final_score'],
                    color=TIER_COLORS[tier], s=90, label=tier, zorder=4,
                    edgecolors='white', linewidth=0.8)
        for _, row in group.iterrows():
            ax6.annotate(row['name'].split()[0], (row['years_exp'], row['final_score']),
                         fontsize=7, color=COLORS['mid'],
                         xytext=(4, 3), textcoords='offset points')

    ax6.set_facecolor(COLORS['bg'])
    ax6.spines[['top', 'right']].set_visible(False)
    ax6.spines[['left', 'bottom']].set_color(COLORS['grid'])
    ax6.tick_params(colors=COLORS['mid'], labelsize=9)
    ax6.set_title('Experience vs Fit Score', fontsize=11, fontweight='bold',
                  color=COLORS['dark'], pad=10)
    ax6.set_xlabel('Years of Experience', fontsize=9, color=COLORS['mid'])
    ax6.set_ylabel('Fit Score (%)', fontsize=9, color=COLORS['mid'])
    ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax6.grid(color=COLORS['grid'], linewidth=0.5, linestyle='--')
    ax6.legend(fontsize=8, framealpha=0.9)

    path = f'/mnt/user-data/outputs/screening_dashboard_{job_key.replace(" ", "_").lower()}.png'
    plt.savefig(path, dpi=155, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Dashboard saved → {path.split('/')[-1]}")
    return path


# ── 8. CANDIDATE REPORT ───────────────────────────────────────────────────────

def print_candidate_report(ranked, job_key, top_n=5):
    print(f"\n{'='*65}")
    print(f"  SCREENING REPORT — {job_key}")
    print(f"{'='*65}")
    print(f"  Total candidates assessed: {len(ranked)}")
    print(f"  Strong Match:  {(ranked['tier'] == 'Strong Match').sum()}")
    print(f"  Potential:     {(ranked['tier'] == 'Potential').sum()}")
    print(f"  Weak Match:    {(ranked['tier'] == 'Weak Match').sum()}")
    print(f"\n  Top {top_n} Candidates:\n")

    for _, row in ranked.head(top_n).iterrows():
        print(f"  #{row['rank']}  {row['name']}  ({row['years_exp']} yrs exp)")
        print(f"      Score:      {row['final_score']:.1f}%  [{row['tier']}]")
        print(f"      Skills:     {row['skill_score']:.1f}%  |  Semantic: {row['semantic_score']:.1f}%")
        print(f"      Matched:    {', '.join(row['req_matched']) or 'none'}")
        if row['req_missing']:
            print(f"      Missing:    {', '.join(row['req_missing'])}")
        print()


# ── 9. EXPORT ─────────────────────────────────────────────────────────────────

def export_results(df, all_ranked):
    df_export = df[['candidate_id', 'name', 'years_exp', 'current_role']].copy()
    df_export['skills'] = df['skills'].apply(lambda x: ', '.join(x))
    df_export.to_csv('/mnt/user-data/outputs/candidates.csv', index=False)

    for job_key, ranked in all_ranked.items():
        fname = f'/mnt/user-data/outputs/ranked_{job_key.replace(" ", "_").lower()}.csv'
        ranked[['rank', 'name', 'years_exp', 'tier', 'final_score',
                'skill_score', 'semantic_score', 'match_pct_req']].to_csv(fname, index=False)

    print("  CSVs exported.")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Building candidate pool...")
    df = build_dataset()
    print(f"  {len(df)} candidates loaded")

    all_ranked = {}
    for job_key in JOB_DESCRIPTIONS:
        print(f"\nScreening for: {job_key}")
        ranked = rank_candidates(df, job_key)
        all_ranked[job_key] = ranked
        plot_dashboard(ranked, job_key)
        print_candidate_report(ranked, job_key, top_n=5)

    export_results(df, all_ranked)
    print("\nDone. All outputs saved to outputs/")
