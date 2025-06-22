import os
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import classification_report

from utils import load_dataframe, prepare_dataset
from model_trainer import train_model, MusicGenreModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_MODELS = 10
ENSEMBLE_DIR = 'data/ensemble_models'
ENSEMBLE_PRED_PATH = 'data/ensemble_genre_predictions.tsv'
ENSEMBLE_REPORT_PATH = 'data/ensemble_report.txt'

os.makedirs(ENSEMBLE_DIR, exist_ok=True)

def run_ensemble_training():
    print("Loading dataset...")
    df = load_dataframe(tag_type='genre')
    X, y, mlb, scaler, _ = prepare_dataset(df, 'main_genres')
    input_dim = X.shape[1]
    num_classes = y.shape[1]
    X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    y_np = y

    all_preds = []

    for i in range(N_MODELS):
        print(f"\n=== Training model {i+1}/{N_MODELS} ===")
        model_path = os.path.join(ENSEMBLE_DIR, f'model_{i+1}.pt')
        mlb_path = os.path.join(ENSEMBLE_DIR, f'mlb_{i+1}.pkl')
        scaler_path = os.path.join(ENSEMBLE_DIR, f'scaler_{i+1}.pkl')

        train_model(X, y, mlb, model_path=model_path, mlb_path=mlb_path, scaler_path=scaler_path, scaler=scaler)

        print(f"Evaluating model {i+1}...")
        model = MusicGenreModel(input_dim, num_classes).to(DEVICE)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        with torch.no_grad():
            y_pred = model(X_tensor).cpu().numpy()
        all_preds.append(y_pred)

    print("\n=== Averaging predictions ===")
    avg_preds = np.mean(all_preds, axis=0)

    print("Loading thresholds from final model...")
    thresholds = joblib.load(os.path.join(ENSEMBLE_DIR, f'genre_thresholds.pkl'))
    threshold_arr = np.array([thresholds[g] for g in mlb.classes_])
    y_pred_bin = (avg_preds > threshold_arr).astype(int)

    print("Generating classification report...")
    report = classification_report(y_np, y_pred_bin, target_names=mlb.classes_, zero_division=0, digits=3)
    print(report)

    with open(ENSEMBLE_REPORT_PATH, 'w') as f:
        f.write(report)

    print(f"Saved ensemble report to {ENSEMBLE_REPORT_PATH}")

    print("Saving detailed predictions...")
    pred_tags = mlb.inverse_transform(y_pred_bin)
    true_tags = mlb.inverse_transform(y_np)

    results_df = pd.DataFrame({
        'true_genres': [", ".join(tags) for tags in true_tags],
        'predicted_genres': [", ".join(tags) for tags in pred_tags]
    })

    if 'filename' in df.columns:
        results_df['filename'] = df['filename']

    results_df.to_csv(ENSEMBLE_PRED_PATH, sep='\t', index=False)
    print(f"Saved ensemble predictions to {ENSEMBLE_PRED_PATH}")

if __name__ == "__main__":
    run_ensemble_training()
