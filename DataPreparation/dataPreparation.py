import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# ETL PIPELINE - DATA PREPARATION
# ============================================================================

def etl_pipeline(path_combined, path_djia, path_reddit, output_path):
    """
    ETL Pipeline për pastrimin dhe bashkimin e dataseteve
    
    Returns:
    - df_merged: Merged dataset (i pastër dhe i bashkuar)
    - df_reddit: Reddit News dataset (i pastër)
    - df_train: Training dataset (<2015)
    - df_test: Testing dataset (>=2015)
    """
    
    print("="*80)
    print("ETL PIPELINE - DATA PREPARATION")
    print("="*80)
    
    # EXTRACT
    print("\n[EXTRACT] Ngarkimi i dataseteve...")
    df1 = pd.read_csv(path_combined)
    df2 = pd.read_csv(path_djia)
    df3 = pd.read_csv(path_reddit)
    print(f"[OK] Dataset 1: {df1.shape} | Dataset 2: {df2.shape} | Dataset 3: {df3.shape}")
    
    # TRANSFORM
    print("\n[TRANSFORM] Pastrimi i dataseteve...")
    
    # Dataset 1
    df1['Date'] = pd.to_datetime(df1['Date'])
    
    def clean_news_text(text):
        if pd.isna(text):
            return ''
        text = str(text)
        if text.startswith("b'"):
            text = text[2:]
        elif text.startswith('b"'):
            text = text[2:]
        if text.endswith("'"):
            text = text[:-1]
        elif text.endswith('"'):
            text = text[:-1]
        return text.strip()
    
    top_cols = [col for col in df1.columns if col.startswith('Top')]
    for col in top_cols:
        df1[col] = df1[col].fillna("")
        df1[col] = df1[col].apply(clean_news_text)
    
    # Dataset 2
    df2['Date'] = pd.to_datetime(df2['Date'])
    
    # Dataset 3
    df3['Date'] = pd.to_datetime(df3['Date'])
    df3 = df3.drop_duplicates()
    df_reddit = df3.copy()
    
    print("[OK] Të gjithë datasetet u pastruan")
    
    # LOAD & MERGE
    print("\n[LOAD] Bashkimi i dataseteve...")
    df_merged = pd.merge(df1, df2, on='Date', how='inner')
    print(f"[OK] Merged shape: {df_merged.shape}")
    
    # TRAIN/TEST SPLIT
    print("\n[SPLIT] Train/Test split (<2015 vs >=2015)...")
    df_merged['Year'] = df_merged['Date'].dt.year
    
    df_train = df_merged[df_merged['Year'] < 2015].copy()
    df_test = df_merged[df_merged['Year'] >= 2015].copy()
    
    print(f"[OK] TRAIN: {len(df_train)} rreshta | TEST: {len(df_test)} rreshta")
    
    # SAVE
    print("\n[SAVE] Ruajtja e dataseteve në cleanedData...")
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    df_merged.to_csv(output_path / "Combined_News_DJIA_Merged.csv", index=False)
    df_reddit.to_csv(output_path / "RedditNews_cleaned.csv", index=False)
    df_train.to_csv(output_path / "df_train.csv", index=False)
    df_test.to_csv(output_path / "df_test.csv", index=False)
    
    print("[OK] Të 4 datasetet u ruajtën")
    
    print("\n" + "="*80)
    print("[OK] ETL PIPELINE PËRFUNDUAR ME SUKSES!")
    print("="*80)
    
    return df_merged, df_reddit, df_train, df_test

def _default_paths():
    here = Path(__file__).resolve().parent
    raw_dir = here / "datasets" / "rawData"
    cleaned_dir = here / "datasets" / "cleanedData"
    return {
        "path_combined": raw_dir / "Combined_News_DJIA.csv",
        "path_djia": raw_dir / "upload_DJIA_table.csv",
        "path_reddit": raw_dir / "RedditNews.csv",
        "output_path": cleaned_dir,
    }


if __name__ == "__main__":
    paths = _default_paths()
    df_merged, df_reddit, df_train, df_test = etl_pipeline(
        path_combined=paths["path_combined"],
        path_djia=paths["path_djia"],
        path_reddit=paths["path_reddit"],
        output_path=paths["output_path"],
    )