# Analiza e Lajmeve Financiare dhe Lëvizjes së Aksioneve në Python

Ky projekt analizon nëse lajmet financiare kanë lidhje me lëvizjen ditore të tregut (DJIA), duke përdorur pastrim të të dhënave, krahasim statistikor dhe vizualizime në Python (pa machine learning).

## Çfarë përfshin
- **Importimi i dataset-it** (Kaggle)
- **Pastrimi dhe përgatitja e të dhënave** (ETL)
- **Bashkimi i lajmeve me çmimet e DJIA**
- **Analizë statistikore bazike** (krahasim i kthimeve ditore sipas `Label`)
- **Vizualizime me grafikë**
- **Përfundime & interpretim**

## Struktura
- `DataPreparation/`
  - `dataPreparation.py` – ETL pipeline (lexim, pastrim, merge, train/test split)
  - `datasets/rawData/` – dataset-et origjinale
  - `datasets/cleanedData/` – dataset-et e pastruara / të bashkuara
- `analysis.py` – analiza + gjenerim grafikësh në `outputs/`

## Si ta ekzekutosh

Instalo paketat:

```bash
pip install -r requirements.txt
```

1) ETL (krijon/mbishkruan dataset-et te `DataPreparation/datasets/cleanedData/`):

```bash
python DataPreparation/dataPreparation.py
```

2) Analiza & grafiqe (krijon `outputs/`):

```bash
python analysis.py
```

Do krijohen:
- `outputs/djia_close_timeseries.png`
- `outputs/daily_returns_by_label.png`
- `outputs/summary.txt`

## Streamlit UI (kohë reale)

Për një UI më profesionale ku grafiqet rifreskohen në kohë reale me filtra:

```bash
streamlit run app.py
```

## Dataset
Dataset-i i përdorur është i tipit “Combined News DJIA” (lajme + DJIA table). Në këtë repo janë vendosur në:
- `DataPreparation/datasets/rawData/`

Nëse i shkarkon vetë nga Kaggle, thjesht sigurohu që emrat e skedarëve të jenë:
- `Combined_News_DJIA.csv`
- `upload_DJIA_table.csv`
- `RedditNews.csv`

