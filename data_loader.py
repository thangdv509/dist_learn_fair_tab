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
import json


def format_sentence_for_bert(text, remove_stop_words=False, normalize_spacing=True):
    stop_words = {'a', 'an', 'the', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
                  'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}

    if remove_stop_words:
        words = text.split()
        text = ' '.join([w for w in words if (w.lower() not in stop_words) or (w.lower() in {'and', 'or'})])

    if normalize_spacing:
        # ---- IMPORTANT: preserve raw categorical values (do NOT split / or operators) ----
        # Keep compound operators as-is (optional; safe)
        text = re.sub(r'!\s*=', '!=', text)
        text = re.sub(r'<\s*=', '<=', text)
        text = re.sub(r'>\s*=', '>=', text)

        # Punctuation: remove spaces BEFORE, ensure one space AFTER
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])(?=\S)', r'\1 ', text)

        # Collapse spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Final guarantee (just in case)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)


    return text



def load_sentence_template(dataset_name, template_file="sentence_templates.json"):
    """
    Load sentence template from JSON file.
    
    Args:
        dataset_name: Name of dataset (e.g., 'german-credit-data', 'adult')
        template_file: Path to JSON template file
    
    Returns:
        dict with 'sentences' (list of template strings) and 'field_mapping' (dict)
    """
    if not os.path.exists(template_file):
        raise FileNotFoundError(f"Template file not found: {template_file}")
    
    with open(template_file, 'r', encoding='utf-8') as f:
        templates = json.load(f)
    
    if dataset_name not in templates:
        raise ValueError(f"Dataset '{dataset_name}' not found in template file. Available: {list(templates.keys())}")
    
    return templates[dataset_name]


def format_sentence_from_template(row, template_config):
    """
    Format a sentence from template using row data.
    
    Args:
        row: pandas Series or dict with data
        template_config: dict with 'sentences' and 'field_mapping' from load_sentence_template
    
    Returns:
        Formatted sentence string
    """
    def get_val(field_name):
        """Get value from row using field mapping variants."""
        variants = template_config['field_mapping'].get(field_name, [field_name])
        for variant in variants:
            val = row.get(variant)
            if pd.notna(val) and str(val).strip():
                return str(val).strip()
        return None
    
    # Extract all values and normalize special values
    def normalize_value(value, field_name):
        """Normalize special values for more natural language."""
        if not value:
            return None
        
        # Handle num_people - singular/plural (check first before string conversion)
        if field_name == 'num_people':
            try:
                num = int(float(value))
                if num == 1:
                    return '1 dependent'
                else:
                    return f'{num} dependents'
            except (ValueError, TypeError):
                pass
        
        # Handle existing_credits - singular/plural (check before string conversion)
        if field_name == 'existing_credits':
            try:
                num = int(float(value))
                if num == 1:
                    return '1'
                else:
                    return str(num)
            except (ValueError, TypeError):
                pass
        
        value_str = str(value).strip().lower()
        
        # Handle boolean-like values
        if value_str in ['yes', 'y', '1', 'true']:
            if field_name == 'telephone':
                return 'have'
            elif field_name == 'foreign_worker':
                return 'a'
            return str(value).strip()  # Keep original for other fields
        
        if value_str in ['no', 'n', '0', 'false', 'none']:
            if field_name == 'telephone':
                return 'have no'
            elif field_name == 'foreign_worker':
                return 'not a'
            elif field_name == 'other_debtors' and value_str == 'none':
                return 'no'
            elif field_name == 'other_installment' and value_str == 'none':
                return 'no'
            return str(value).strip()  # Keep original for other fields
        
        return str(value).strip()  # Return original value
    
    values = {}
    for field_name in template_config['field_mapping'].keys():
        raw_value = get_val(field_name)
        values[field_name] = normalize_value(raw_value, field_name) if raw_value else None
    
    # Format each sentence template
    formatted_sentences = []
    for template in template_config['sentences']:
        # Replace placeholders with values
        formatted = template
        for field_name, value in values.items():
            placeholder = f"{{{field_name}}}"
            if value:
                formatted = formatted.replace(placeholder, value)
            else:
                # Remove placeholder and clean up surrounding text
                # Handle cases like "with {marital_status} status" -> "with status" -> remove "with "
                # Handle cases like "checking is {checking_account} and savings is {savings_account}" 
                formatted = re.sub(rf'\s*{re.escape(placeholder)}\s*', ' ', formatted)
        
        # Clean up patterns like "with  status" -> remove "with "
        formatted = re.sub(r'\s+with\s+status', '', formatted, flags=re.IGNORECASE)
        formatted = re.sub(r'\s+and\s+and\s+', ' and ', formatted)  # Fix double "and"
        formatted = re.sub(r'\s+have\s+have\s+', ' have ', formatted)  # Fix double "have"
        formatted = re.sub(r'\s*,\s*,', ',', formatted)
        formatted = re.sub(r'\s*;\s*;', ';', formatted)
        
        # Fix "1 existing credit account" vs "2 existing credit account" -> add 's' for plural
        formatted = re.sub(r'(\d+)\s+existing credit account(?!s)', 
                          lambda m: f"{m.group(1)} existing credit account{'s' if int(m.group(1)) != 1 else ''}", 
                          formatted)
        
        # Clean up extra spaces
        formatted = re.sub(r'\s+', ' ', formatted)
        formatted = formatted.strip()
        
        # Remove leading/trailing punctuation issues
        formatted = re.sub(r'^\s*[,;]\s*', '', formatted)
        formatted = re.sub(r'\s*[,;]\s*$', '', formatted)
        
        if formatted and len(formatted) > 1:
            formatted_sentences.append(formatted)
    
    # Join sentences with space
    result = " ".join(formatted_sentences)
    
    # Final cleanup
    result = re.sub(r'\s+', ' ', result)
    result = result.strip()
    
    return result


def format_adult_sentence(row, columns):
    """
    Format Adult dataset row into natural language sentence.
    Uses common template, includes ALL attributes, keeps original values unchanged.
    """
    # Get all columns except label/target columns
    exclude_cols = {'label', 'class-label', 'class_label', 'target', '_sentence', '_sentence_index'}
    
    # Extract values (keep original, unchanged)
    age = row.get('age')
    gender = row.get('gender') or row.get('sex')
    race = row.get('race')
    marital_status = row.get('marital-status') or row.get('marital_status')
    relationship = row.get('relationship')
    workclass = row.get('workclass')
    education = row.get('education')
    educational_num = row.get('educational-num') or row.get('educational_num')
    occupation = row.get('occupation')
    hours_per_week = row.get('hours-per-week') or row.get('hours_per_week')
    capital_gain = row.get('capital-gain') or row.get('capital_gain')
    capital_loss = row.get('capital-loss') or row.get('capital_loss')
    native_country = row.get('native-country') or row.get('native_country')
    fnlwgt = row.get('fnlwgt')
    
    # Build natural sentence using common template
    parts = []
    
    # Demographic section (natural format)
    if pd.notna(age):
        parts.append(f"{age} year old")
    if pd.notna(gender):
        parts.append(str(gender).strip())
    if pd.notna(race):
        parts.append(f"from {str(race).strip()} race")
    if pd.notna(marital_status):
        parts.append(str(marital_status).strip())
    if pd.notna(relationship):
        parts.append(f"relationship {str(relationship).strip()}")
    
    # Work section (natural format)
    work_parts = []
    if pd.notna(occupation):
        work_parts.append(f"works as {str(occupation).strip()}")
    if pd.notna(workclass):
        work_parts.append(f"in {str(workclass).strip()}")
    if pd.notna(education):
        work_parts.append(f"with {str(education).strip()} education")
    if pd.notna(educational_num):
        work_parts.append(f"educational num {str(educational_num).strip()}")
    if pd.notna(hours_per_week):
        work_parts.append(f"working {str(hours_per_week).strip()} hours per week")
    
    # Financial section
    financial_parts = []
    if pd.notna(capital_gain) and str(capital_gain).strip() != '0':
        financial_parts.append(f"capital gain {str(capital_gain).strip()}")
    if pd.notna(capital_loss) and str(capital_loss).strip() != '0':
        financial_parts.append(f"capital loss {str(capital_loss).strip()}")
    
    # Other attributes
    other_parts = []
    if pd.notna(native_country):
        other_parts.append(f"from {str(native_country).strip()}")
    if pd.notna(fnlwgt):
        other_parts.append(f"fnlwgt {str(fnlwgt).strip()}")
    
    # Combine all parts naturally
    sentence_parts = []
    if parts:
        sentence_parts.append(" ".join(parts))
    if work_parts:
        sentence_parts.append(", ".join(work_parts))
    if financial_parts:
        sentence_parts.append(", ".join(financial_parts))
    if other_parts:
        sentence_parts.append(", ".join(other_parts))
    
    sentence = ", ".join(sentence_parts) if sentence_parts else "person"
    
    # Clean up spacing
    sentence = re.sub(r'\s+,', ',', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r',\s*,', ',', sentence)
    sentence = sentence.strip()
    sentence = re.sub(r',([^\s])', r', \1', sentence)
    
    return sentence


def format_german_credit_sentence(row, columns):
    """
    Format German Credit dataset row into natural language sentence.
    Uses template from sentence_templates.json file.
    """
    try:
        template_config = load_sentence_template('german-credit-data')
        sentence = format_sentence_from_template(row, template_config)
        return sentence
    except (FileNotFoundError, ValueError, KeyError) as e:
        # Fallback to old method if template file not found
        print(f"Warning: Could not load template, using fallback: {e}")
        # Simple fallback
        age = row.get('age') or row.get('Age')
        sex = row.get('sex') or row.get('Sex') or row.get('gender') or row.get('Gender')
        marital_status = row.get('marital-status') or row.get('marital_status') or row.get('Marital-Status')
        job = row.get('job') or row.get('Job') or row.get('occupation') or row.get('Occupation')
        credit_amount = row.get('credit-amount') or row.get('credit_amount') or row.get('Credit-Amount')
        duration = row.get('duration') or row.get('Duration')
        purpose = row.get('purpose') or row.get('Purpose')
        checking_account = row.get('checking-account') or row.get('checking_account') or row.get('Checking-Account')
        savings_account = row.get('savings-account') or row.get('savings_account') or row.get('Savings-Account')
        
        parts = []
        if age:
            parts.append(f"A {age}-year-old")
        if sex:
            parts.append(str(sex))
        parts.append("applicant")
        if marital_status:
            parts.append(f"with {marital_status} status")
        if job:
            parts.append(f"working as {job}")
        
        sentence1 = " ".join(parts) + "."
        
        parts2 = []
        if credit_amount or purpose or duration:
            credit_info = []
            if credit_amount:
                credit_info.append(f"seek {credit_amount}")
            if purpose:
                credit_info.append(f"for {purpose}")
            if duration:
                credit_info.append(f"across {duration} months")
            if credit_info:
                parts2.append("They " + " ".join(credit_info))
        
        if checking_account or savings_account:
            account_info = []
            if checking_account:
                account_info.append(f"checking is {checking_account}")
            if savings_account:
                account_info.append(f"savings is {savings_account}")
            if account_info:
                if parts2:
                    parts2.append("; their " + " and ".join(account_info))
                else:
                    parts2.append("Their " + " and ".join(account_info))
        
        sentence2 = "".join(parts2) + "." if parts2 else ""
        
        result = sentence1 + " " + sentence2 if sentence2 else sentence1
        return re.sub(r'\s+', ' ', result).strip()


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
    
    # Convert to natural language sentences with improved formatting
    sentences = []
    for idx, row in df.iterrows():
        # Use improved format_german_credit_sentence function
        sentence = format_german_credit_sentence(row, df.columns)
        
        # Preprocess to improve tokenization
        sentence = format_sentence_for_bert(sentence)
        
        sentences.append(sentence)
    
    labels = df['label'].tolist()
    
    # Store mapping in df for reverse lookup (sentence index -> original row)
    # This allows matching sentences back to original table rows
    df['_sentence'] = sentences
    df['_sentence_index'] = range(len(sentences))
    df = df.reset_index(drop=True)
    
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
            
            # Store mapping in df for reverse lookup
            df['_sentence'] = sentences
            df['_sentence_index'] = range(len(sentences))
            df = df.reset_index(drop=True)
            
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
    
    # Store mapping in df for reverse lookup
    data['_sentence'] = sentences
    data['_sentence_index'] = range(len(sentences))
    
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
    # Keep mapping from sentence index to original row index for reverse lookup
    sentences = []
    sentence_to_row_index = []  # Map: sentences[i] corresponds to df.iloc[sentence_to_row_index[i]]
    
    for idx, row in df.iterrows():
        # Special handling for specific datasets
        if dataset_name == 'adult':
            sentence = format_adult_sentence(row, df.columns)
        elif dataset_name == 'german-credit-data' or 'german' in dataset_name.lower():
            sentence = format_german_credit_sentence(row, df.columns)
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
        sentence_to_row_index.append(idx)  # Store original row index
    
    # Create reverse mapping DataFrame for easy lookup
    mapping_df = pd.DataFrame({
        'sentence_index': range(len(sentences)),
        'original_row_index': sentence_to_row_index,
        'sentence': sentences,
        'label': df.loc[sentence_to_row_index, 'label'].values
    })
    
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