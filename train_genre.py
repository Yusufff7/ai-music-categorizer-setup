import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import load_dataframe, prepare_dataset
from model_trainer import train_model

def main():
    df = load_dataframe(tag_type='genre')
    print("Class distribution in training data:")
    print(pd.Series([g for sublist in df['main_genres'] for g in sublist]).value_counts())
    X, y, mlb, scaler, y_raw = prepare_dataset(df, 'main_genres')

    train_model(
        X, y, mlb,
        model_path=os.path.join('data', 'genre_model.pt'),
        mlb_path=os.path.join('data', 'mlb_genre.pkl'),
        scaler_path=os.path.join('data', 'scaler_genre.pkl'),
        scaler=scaler
    )


if __name__ == "__main__":
    main()