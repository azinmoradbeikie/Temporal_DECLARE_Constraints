import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def normalize_key(s):
    """Normalize constraint key format for matching."""
    s = s.strip('<>')  # Remove angle brackets
    s = re.sub(r'\s+', '', s).lower()
    if not s.endswith(']'):
        s += ']'
    if not s.startswith('<'):
        s = '<' + s
    return s


def preprocess_constraint(row):
    """Convert a constraint to a single 'word' format and store mapping."""
    template_raw = str(row[1]).replace(' ', '')  # template
    try:
        activities_list = eval(row[2])  # activities (assumed as list-like string)
    except:
        activities_list = []
    activities_clean = [act.replace(' ', '') for act in activities_list]
    constraint_word = f"{template_raw}[{','.join(activities_clean)}]"

    word_to_template_activities[constraint_word] = (row[1], activities_list)
    return constraint_word

# Dictionary for mapping constraint words to (template, activities)
word_to_template_activities = {}

def rank_constraints(input_path, output_path):
    """Run the TF-IDF ranking process"""
    df = pd.read_csv(input_path, sep=',')
    
    # If unprocessed, preprocess into constraint strings
    if len(df.columns) > 2:
        df['constraint_str'] = df.apply(preprocess_constraint, axis=1)
        df = df[[df.columns[0], 'constraint_str']]
        df.columns = ['community_id', 'constraint_str']
    else:
        df.columns = ['community_id', 'constraint_str']
    
    # TF-IDF processing
    full_doc = ' '.join(df['constraint_str'].tolist())
    vectorizer = TfidfVectorizer(token_pattern=r"[^ ]+")
    X = vectorizer.fit_transform([full_doc])
    
    # Create results DataFrame
    feature_names = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]
    
    tfidf_df = pd.DataFrame({
        'constraint_word': feature_names,
        'tfidf_score': scores
    })
    
    # Normalize and map back to original templates
    normalized_word_to_template_activities = {
        normalize_key(k): v for k, v in word_to_template_activities.items()
    }
    
    tfidf_df['mapped'] = tfidf_df['constraint_word'].apply(
        lambda x: normalized_word_to_template_activities.get(normalize_key(x), None)
    )
    
    mapped = tfidf_df['mapped']
    tfidf_df['template'] = mapped.apply(lambda x: x[0] if x else None)
    tfidf_df['activities'] = mapped.apply(lambda x: x[1] if x else None)
    
    # Drop unmatched entries and merge with community IDs
    tfidf_df = tfidf_df.dropna(subset=['template', 'activities'])
    df['key'] = df['constraint_str'].apply(normalize_key)
    tfidf_df['key'] = tfidf_df['constraint_word'].apply(normalize_key)
    tfidf_df = tfidf_df.merge(df[['key', 'community_id']], on='key', how='left')
    
    # Final output
    tfidf_df = tfidf_df[['community_id', 'template', 'activities', 'tfidf_score']]
    tfidf_df = tfidf_df.sort_values(by='tfidf_score', ascending=False)
    tfidf_df.to_csv(output_path, index=False)
    
    print(f"Ranked constraints saved to {output_path}")
    return tfidf_df