"""
Generate synthetic German Credit Data CSV for testing.
Run this file to create german_credit_data.csv
"""

import pandas as pd
import numpy as np

def create_synthetic_german_credit_data(n_samples=500, random_state=42):
    """
    Create synthetic German Credit Data similar to the original dataset.
    """
    np.random.seed(random_state)
    
    data = {
        'age': np.random.randint(18, 75, n_samples),
        'sex': np.random.choice(['male', 'female'], n_samples),
        'credit_amount': np.random.randint(100, 20000, n_samples),
        'duration': np.random.randint(1, 72, n_samples),
        'purpose': np.random.choice(
            ['car', 'furniture', 'education', 'business', 'domestic appliance', 'repairs', 'other'],
            n_samples
        ),
        'job': np.random.choice(
            ['unemployed', 'unskilled', 'skilled', 'management'],
            n_samples
        ),
        'checking_account': np.random.choice(
            ['no account', 'negative balance', 'positive balance', 'no checking account'],
            n_samples
        ),
        'savings_account': np.random.choice(
            ['little', 'moderate', 'quite rich', 'rich', 'no savings'],
            n_samples
        ),
    }
    
    df = pd.DataFrame(data)
    
    # Generate labels based on features (simple rule-based)
    # Higher age, higher credit amount (up to a point), longer duration slightly negative
    score = (
        (df['age'] >= 30) * 0.3 +
        (df['age'] >= 40) * 0.2 +
        (df['credit_amount'] < 10000) * 0.3 +
        (df['duration'] <= 24) * 0.2 +
        (df['sex'] == 'male') * 0.1 +
        (df['job'].isin(['management', 'skilled'])) * 0.3 +
        (df['checking_account'].isin(['positive balance'])) * 0.2 +
        (df['savings_account'].isin(['quite rich', 'rich'])) * 0.3
    )
    
    # Add some randomness
    score += np.random.normal(0, 0.2, n_samples)
    
    # Convert to binary labels
    df['class'] = ['good' if s > 1.5 else 'bad' for s in score]
    
    return df

if __name__ == "__main__":
    print("Generating synthetic German Credit Data...")
    df = create_synthetic_german_credit_data(n_samples=500)
    
    # Save to CSV
    output_file = 'german_credit_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Saved {len(df)} samples to {output_file}")
    print(f"\nDataset info:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"\nLabel distribution:")
    print(df['class'].value_counts())
    print(f"\nSample rows:")
    print(df.head(3))
