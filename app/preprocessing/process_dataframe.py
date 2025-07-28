import pandas as pd

from app.preprocessing.process_category import normalize_name_category


def remove_empty_values(df: pd.DataFrame) -> pd.DataFrame:
    
    df_with_empty_rows = df[
    (df['headline'].isna()) | (df['category'].isna()) |
    (df['headline'] == '')  | (df['category'] == '')
    ]

    df_cleaned = df.drop(df_with_empty_rows.index)

    return df_cleaned

def remove_duplicated(df: pd.DataFrame) -> pd.DataFrame:
    
    df_cleaned = df.drop_duplicates()
    return df_cleaned

def normalize_category(df: pd.DataFrame) -> pd.DataFrame:
    df['normalized_category'] = df['category'].apply(normalize_name_category)
    df['category'] = df['normalized_category']
    df.drop(columns=['normalized_category'], inplace=True)
    
    return df


# NOTA: Esta función está diseñada únicamente para fines de entrenamiento rápido
def minimize_df(df: pd.DataFrame, n=10) -> pd.DataFrame:

    #Reduce el DataFrame a las 10 categorías menos frecuentes
    #Se usa para probar inicialmente el tiempo que tarda el modelo en entrenarse,
    #especialmente útil en entornos con recursos limitados.


    category_counts = df['category'].value_counts()

    #10 categorias con menor cantidad de rows
    rarest_categories = category_counts.nsmallest(n).index.tolist()

    # filtrar
    minimized_df = df[df['category'].isin(rarest_categories)].reset_index(drop=True)
    return minimized_df