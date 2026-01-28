"""
Load German Credit Data and convert to canonical natural language sentences.
Sensitive attributes: age, sex
Task: predict loan approval (credit risk)

Download link: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
Or use local CSV file: german_credit_data.csv
"""

import pandas as pd
import numpy as np
import os
import re


def format_sentence_for_bert(text):
    """
    Preprocess sentence to improve BERT tokenization.
    - Adds spaces around punctuation to avoid splitting
    - Formats numbers better (keeps them as single tokens when possible)
    - Removes unnecessary punctuation
    - Handles special characters like <, >, =, etc.
    """
    # Add spaces around operators and special characters
    text = re.sub(r'([<>=])(?=\S)', r'\1 ', text)  # Add space after <, >, =
    text = re.sub(r'(?<=\S)([<>=])', r' \1', text)  # Add space before <, >, =
    
    # Add spaces around common punctuation (but not if already spaced)
    text = re.sub(r'([.,;:!?/])(?=\S)', r'\1 ', text)  # Add space after punctuation
    text = re.sub(r'(?<=\S)([.,;:!?/])', r' \1', text)  # Add space before punctuation
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing spaces
    text = text.strip()
    
    return text


def format_adult_sentence(row, columns):
    """
    Format Adult dataset row into natural language sentence.
    Better formatting for demographic and task-relevant information.
    """
    # Extract key attributes
    age = int(row.get('age', 0)) if pd.notna(row.get('age')) else 0
    gender = str(row.get('gender', row.get('sex', 'unknown'))).lower().strip() if pd.notna(row.get('gender', row.get('sex'))) else 'unknown'
    race = str(row.get('race', 'unknown')).lower().strip() if pd.notna(row.get('race')) else 'unknown'
    marital_status = str(row.get('marital-status', row.get('marital_status', 'unknown'))).lower().strip() if pd.notna(row.get('marital-status', row.get('marital_status'))) else 'unknown'
    relationship = str(row.get('relationship', 'unknown')).lower().strip() if pd.notna(row.get('relationship')) else 'unknown'
    
    workclass = str(row.get('workclass', 'unknown')).lower().strip() if pd.notna(row.get('workclass')) else 'unknown'
    education = str(row.get('education', 'unknown')).lower().strip() if pd.notna(row.get('education')) else 'unknown'
    occupation = str(row.get('occupation', 'unknown')).lower().strip() if pd.notna(row.get('occupation')) else 'unknown'
    hours_per_week = int(row.get('hours-per-week', row.get('hours_per_week', 0))) if pd.notna(row.get('hours-per-week', row.get('hours_per_week'))) else 0
    
    capital_gain = int(row.get('capital-gain', row.get('capital_gain', 0))) if pd.notna(row.get('capital-gain', row.get('capital_gain'))) else 0
    capital_loss = int(row.get('capital-loss', row.get('capital_loss', 0))) if pd.notna(row.get('capital-loss', row.get('capital_loss'))) else 0
    native_country = str(row.get('native-country', row.get('native_country', 'unknown'))).lower().strip() if pd.notna(row.get('native-country', row.get('native_country'))) else 'unknown'
    
    # Format: Demographic section first (should be picked up by z_d)
    demographic_parts = []
    if age > 0:
        demographic_parts.append(f"{age} year old")
    if gender != 'unknown':
        demographic_parts.append(gender)
    if race != 'unknown' and race != 'white':  # Only mention if not default
        demographic_parts.append(f"{race} race")
    if marital_status != 'unknown':
        demographic_parts.append(f"marital status {marital_status}")
    if relationship != 'unknown':
        demographic_parts.append(f"relationship {relationship}")
    
    demographic_section = " ".join(demographic_parts) if demographic_parts else "person"
    
    # Task-relevant section (employment, education, income indicators)
    task_parts = []
    if workclass != 'unknown':
        task_parts.append(f"workclass {workclass}")
    if education != 'unknown':
        task_parts.append(f"education {education}")
    if occupation != 'unknown':
        task_parts.append(f"occupation {occupation}")
    if hours_per_week > 0:
        task_parts.append(f"works {hours_per_week} hours per week")
    if capital_gain > 0:
        task_parts.append(f"capital gain {capital_gain}")
    if capital_loss > 0:
        task_parts.append(f"capital loss {capital_loss}")
    if native_country != 'unknown' and native_country != 'united-states':
        task_parts.append(f"from {native_country}")
    
    task_section = " ".join(task_parts) if task_parts else ""
    
    # Combine into natural sentence
    if task_section:
        sentence = f"{demographic_section} {task_section}"
    else:
        sentence = demographic_section
    
    return sentence


def load_german_credit_from_csv(csv_path=None):
    """
    Load German Credit Data from local CSV file.
    
    If csv_path is not provided, tries to find it in current directory.
    File should have columns: age, sex, credit_amount, duration, purpose, job, etc.
    
    Returns:
        sentences: List of canonical natural language sentences
        labels: List of binary labels (1 = good, 0 = bad)
        data: Original dataframe
        sensitive_attrs: Dict of sensitive attributes
    """
    if csv_path is None:
        # Look for CSV file in current directory
        possible_paths = [
            'german_credit_data.csv',
            'german.csv',
            'german_credit.csv'
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            raise FileNotFoundError(
                "No German Credit Data CSV file found. "
                "Please provide a CSV file with columns: "
                "age, sex, credit_amount, duration, purpose, job, checking_account, savings_account, class"
            )
    
    print(f"Loading German Credit Data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Normalize column names
    df.columns = [col.lower().strip() for col in df.columns]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Handle target variable (may be named 'class' or 'credit')
    target_col = None
    for col in ['class', 'class-label', 'class_label', 'credit', 'target', 'label']:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError(f"Could not find target column in {df.columns.tolist()}")
    
    # Convert target to binary (1 = good, 0 = bad)
    if df[target_col].dtype == 'object' or df[target_col].dtype == 'string':
        df['label'] = (df[target_col].astype(str).str.lower().str.strip() == 'good').astype(int)
    else:
        # Assume numeric: 1 = good, 0 = bad
        df['label'] = df[target_col].astype(int)
    
    # Handle missing values
    df = df.dropna(subset=['label'])
    
    print(f"After preprocessing: {len(df)} samples")
    print(f"Label distribution: {df['label'].sum()} good, {(1-df['label']).sum()} bad credit")
    
    # Extract sensitive attributes
    sensitive_attrs = {
        'age': df['age'].values if 'age' in df.columns else None,
        'sex': df['sex'].values if 'sex' in df.columns else None
    }
    
    # Convert to natural language sentences
    sentences = []
    for idx, row in df.iterrows():
        age = int(row['age']) if 'age' in row and pd.notna(row['age']) else 0
        # Handle both 'credit_amount' and 'credit-amount'
        credit_col = 'credit-amount' if 'credit-amount' in df.columns else 'credit_amount'
        credit_amount = int(row[credit_col]) if credit_col in row and pd.notna(row[credit_col]) else 0
        
        duration = int(row['duration']) if 'duration' in row and pd.notna(row['duration']) else 0
        sex = str(row['sex']).lower().strip() if 'sex' in row and pd.notna(row['sex']) else 'applicant'
        
        # Handle both 'job' and other variations
        job_col = 'job' if 'job' in df.columns else None
        job = str(row[job_col]).lower().strip() if job_col and job_col in row and pd.notna(row[job_col]) else 'employed'
        
        # Handle both 'checking_account' and 'checking-account'
        checking_col = 'checking-account' if 'checking-account' in df.columns else 'checking_account'
        checking = str(row[checking_col]).lower().strip() if checking_col in row and pd.notna(row[checking_col]) else 'has account'
        
        # Handle both 'savings_account' and 'savings-account'
        savings_col = 'savings-account' if 'savings-account' in df.columns else 'savings_account'
        savings = str(row[savings_col]).lower().strip() if savings_col in row and pd.notna(row[savings_col]) else 'has savings'
        
        purpose = str(row['purpose']).lower().strip() if 'purpose' in row and pd.notna(row['purpose']) else 'other'
        
        # IMPROVED Template: Separate demographic from task content clearly
        # Format numbers and punctuation properly to avoid bad tokenization
        # Add spaces around punctuation and format numbers as words when possible
        
        # Demographic section (age, gender) - should be picked up by z_d
        demographic_section = f"A {age} year old {sex}"
        
        # Job details (task-relevant indicator of creditworthiness)
        employment_section = f"employed as {job}" if job and job != 'employed' else "employed"
        
        # Credit details (task content) - format numbers and currency better
        # Replace "DM" with "marks" to avoid tokenization issues, format numbers as words when possible
        credit_section = f"applying for {credit_amount} marks loan for {purpose} with duration {duration} months"
        
        # Financial status (indicators of creditworthiness) - use "and" instead of comma
        # Format checking and savings better
        checking_clean = checking.replace('dm', 'marks').replace('DM', 'marks')
        savings_clean = savings.replace('dm', 'marks').replace('DM', 'marks')
        financial_section = f"checking account {checking_clean} and savings account {savings_clean}"
        
        # Complete sentence: Demographic first, then task-relevant details
        # Use proper spacing and avoid unnecessary punctuation
        sentence = f"{demographic_section} {employment_section} {credit_section} {financial_section}"
        
        # Preprocess to improve tokenization
        sentence = format_sentence_for_bert(sentence)
        
        sentences.append(sentence)
    
    labels = df['label'].tolist()
    
    return sentences, labels, df, sensitive_attrs


def load_german_credit_data():
    """
    Load German Credit Data (attempts CSV first, then OpenML as fallback).
    """
    try:
        # Try multiple CSV paths
        possible_paths = [
            'data/german-credit-data.csv',
            'data/german_credit_data.csv',
            'data/german_credit.csv',
            'data/german.csv',
            'german-credit-data.csv',
            'german_credit_data.csv',
            'german_credit.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found CSV file at: {path}")
                return load_german_credit_from_csv(csv_path=path)
        
        # If no CSV found, raise error
        raise FileNotFoundError(f"Could not find german credit data CSV. Tried: {possible_paths}")
    except FileNotFoundError as e:
        print(f"\n❌ Local CSV not found: {e}")
        print("\nAttempting to download from OpenML...")
        from sklearn.datasets import fetch_openml
        
        try:
            data = fetch_openml("german_credit_data", version=1, as_frame=True)
            df = data.frame
            
            # Map columns
            df.columns = [col.lower() for col in df.columns]
        
            # Convert to label format
            target_col = 'class'
            df['label'] = (df[target_col].str.lower().str.strip() == 'good').astype(int)
            df = df.dropna(subset=['label'])
            
            print(f"Loaded {len(df)} samples from OpenML")
            
            sensitive_attrs = {
                'age': df['age'].values if 'age' in df.columns else None,
                'sex': df['sex'].values if 'sex' in df.columns else None
            }
            
            # Convert to sentences
            sentences = []
            for idx, row in df.iterrows():
                age = int(row.get('age', 0))
                credit_amount = int(row.get('credit amount', 0))
                duration = int(row.get('duration', 0))
                sex = str(row.get('sex', 'unknown')).lower()
                
                sentence = (
                    f"Applicant age is {age} "
                    f"Gender is {sex} "
                    f"Credit amount is {credit_amount} "
                    f"Duration is {duration} months"
                )
                # Preprocess to improve tokenization
                sentence = format_sentence_for_bert(sentence)
                sentences.append(sentence)
            
            labels = df['label'].tolist()
            return sentences, labels, df, sensitive_attrs
        except Exception as openml_error:
            print(f"❌ OpenML download failed: {openml_error}")
            raise FileNotFoundError(f"Could not load german-credit-data from local CSV or OpenML. Please ensure german-credit-data.csv exists in the data/ folder.")


def load_german_credit_data_balanced(n_samples=None, random_state=42):
    """
    Load German Credit Data and balance the classes.
    
    Args:
        n_samples: Total number of samples to return. If None, use all data
        random_state: Seed for reproducibility
        
    Returns:
        sentences: List of canonical natural language sentences
        labels: List of binary labels
        data: Subset dataframe
        sensitive_attrs: Dict of sensitive attributes
    """
    sentences, labels, data, sensitive_attrs = load_german_credit_data()
    
    # Balance classes
    np.random.seed(random_state)
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    
    # Sample n_samples (if None, use all)
    if n_samples is None:
        selected_indices = indices
    else:
        selected_indices = indices[:min(n_samples, len(indices))]
    
    sentences = [sentences[i] for i in selected_indices]
    labels = [labels[i] for i in selected_indices]
    data = data.iloc[selected_indices].reset_index(drop=True)
    
    sensitive_attrs = {
        'age': data['Age'].values if 'Age' in data.columns else sensitive_attrs['age'],
        'sex': data['Sex'].values if 'Sex' in data.columns else sensitive_attrs['sex']
    }
    
    return sentences, labels, data, sensitive_attrs


def load_dataset_generic(dataset_name, csv_path=None, n_samples=None, random_state=42):
    """
    Generic loader for different datasets.
    Supports: adult, bank-marketing, credit-scoring, dutch-census, german-credit-data, credit-card-clients, kdd-census-income
    
    Args:
        dataset_name: Name of dataset (without .csv)
        csv_path: Path to CSV file (if None, searches in data/ folder)
        n_samples: Number of samples to return (if None, use all)
        random_state: Seed for reproducibility
    
    Returns:
        sentences, labels, data, sensitive_attrs
    """
    import glob
    
    # Find CSV file
    if csv_path is None:
        # Search in data/ folder
        possible_paths = [
            f"data/{dataset_name}.csv",
            f"{dataset_name}.csv",
            f"./data/{dataset_name}.csv"
        ]
        
        # Try glob patterns
        if not any(os.path.exists(p) for p in possible_paths):
            pattern = f"data/*{dataset_name}*"
            matches = glob.glob(pattern)
            if matches:
                csv_path = matches[0]
            else:
                raise FileNotFoundError(f"Could not find CSV for dataset: {dataset_name}")
        else:
            for p in possible_paths:
                if os.path.exists(p):
                    csv_path = p
                    break
    
    print(f"\n{'='*70}")
    print(f"Loading dataset: {dataset_name}")
    print(f"{'='*70}")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {csv_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    
    # Normalize column names
    df.columns = [col.lower().strip() for col in df.columns]
    
    # Find target column (check with and without hyphens)
    target_col = None
    possible_cols = [
        'class', 'target', 'label', 'y', 'outcome', 'risk',
        'class-label', 'class_label', 'target-label', 'target_label'
    ]
    for col in possible_cols:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        raise ValueError(f"Could not find target column. Looking for one of {possible_cols}. Found: {df.columns.tolist()}")
    
    # Convert target to binary
    if df[target_col].dtype == 'object' or df[target_col].dtype == 'string':
        target_str = df[target_col].astype(str).str.lower().str.strip()
        # Handle different label formats (good, yes, >50k, etc.)
        if target_str.str.contains(r'good|yes|true|>50k|high|1\.0', regex=True).any():
            df['label'] = (target_str.str.contains(r'good|yes|true|>50k|high|1\.0', regex=True)).astype(int)
        else:
            # Default: assume numeric-like strings where '1' means positive
            df['label'] = (target_str == '1').astype(int)
    else:
        # Numeric: 1 = positive, 0 = negative
        df['label'] = (df[target_col] == 1).astype(int)
    
    # Drop rows with missing labels
    df = df.dropna(subset=['label'])
    
    # Handle missing values in key columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Sample if needed
    np.random.seed(random_state)
    if n_samples is not None and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=random_state)
    
    # Convert to sentences with better formatting
    sentences = []
    for idx, row in df.iterrows():
        # Special handling for adult dataset
        if dataset_name == 'adult':
            sentence = format_adult_sentence(row, df.columns)
        else:
            # Generic format for other datasets
            attrs = []
            for col in df.columns:
                if col != 'label' and not col.startswith('unnamed'):
                    val = row[col]
                    if pd.notna(val):
                        # Format better: use "is" for cleaner text
                        attrs.append(f"{col} is {val}")
            
            sentence = " ".join(attrs[:10])  # Limit to first 10 attributes
        
        # Preprocess to improve tokenization
        sentence = format_sentence_for_bert(sentence)
        sentences.append(sentence)
    
    labels = df['label'].tolist()
    
    # Extract sensitive attributes (common ones)
    sensitive_attrs = {
        'age': df['age'].values if 'age' in df.columns else None,
        'sex': df['sex'].values if 'sex' in df.columns else (
            df['gender'].values if 'gender' in df.columns else None
        ),
        'race': df['race'].values if 'race' in df.columns else None
    }
    
    print(f"  Total samples: {len(labels)}")
    print(f"  Label distribution: {sum(labels)} (1), {len(labels) - sum(labels)} (0)")
    
    return sentences, labels, df, sensitive_attrs


if __name__ == "__main__":
    # Test loading
    try:
        sentences, labels, data, sensitive_attrs = load_german_credit_data_balanced(n_samples=500)
        
        print("\n" + "="*70)
        print("Sample canonical natural language sentences:")
        print("="*70)
        
        for i in range(min(3, len(sentences))):
            print(f"\n[{i}] {sentences[i]}")
            print(f"    Label: {labels[i]} {'(good credit)' if labels[i] == 1 else '(bad credit)'}")
        
        print(f"\n\nTotal samples: {len(sentences)}")
        print(f"Label distribution: {sum(labels)} good, {len(labels) - sum(labels)} bad")
        
        if sensitive_attrs['sex'] is not None:
            unique_sex = np.unique(sensitive_attrs['sex'])
            print(f"Gender distribution: {unique_sex}")
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease provide a German Credit Data CSV file.")
        print("Download from: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)")
        print("\nThe CSV should have these columns:")
        print("  age, sex, credit_amount, duration, purpose, job, checking_account, savings_account, class")

def save_processed_data(sentences, labels, dataset_name, processed_data_dir="processed_data"):
    """
    Save processed natural language sentences and labels to CSV file.
    
    Args:
        sentences: List of natural language sentences
        labels: List of binary labels
        dataset_name: Name of dataset (e.g., 'german-credit-data', 'adult')
        processed_data_dir: Directory to save processed data (default: 'processed_data')
    """
    import os
    
    # Create processed_data directory if it doesn't exist
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Create output filepath matching original filename
    output_path = os.path.join(processed_data_dir, f"{dataset_name}.csv")
    
    # Create DataFrame with sentences and labels
    df = pd.DataFrame({
        'sentence': sentences,
        'label': labels
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved processed data to: {output_path}")
    print(f"  Samples: {len(sentences)}")
    print(f"  Label distribution: {sum(labels)} (label 1), {len(labels) - sum(labels)} (label 0)")
    
    return output_path