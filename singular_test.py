import os
import numpy as np
import torch
import librosa
import joblib
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import torch.nn as nn
import torch.nn.functional as F

# Saved files
MOOD_MODEL_PATH = 'data/bestMood/mood_model.pt'
MOOD_MLB_PATH = 'data/bestMood/mlb_mood.pkl'
MOOD_SCALER_PATH = 'data/bestMood/scaler_mood.pkl'

GENRE_MODEL_PATH = 'data/bestGenre/genre_model.pt'
GENRE_MLB_PATH = 'data/bestGenre/mlb_genre.pkl'
GENRE_SCALER_PATH = 'data/bestGenre/scaler_genre.pkl'
GENRE_THRESHOLDS_PATH = 'data/bestGenre/genre_thresholds.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MoodClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class MusicGenreModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.4)

        self.attention = nn.Linear(2048, 2048)

        # Genre branches (metal, punk, blues)
        def branch():
            return nn.Sequential(
                nn.Linear(2048, 256),
                nn.SiLU(),
                nn.Linear(256, 256),
                nn.Sigmoid()
            )
        self.metal_branch = branch()
        self.punk_branch = branch()
        self.blues_branch = branch()

        self.final_dense = nn.Linear(2048 + 256*3, 768)
        self.bn2 = nn.BatchNorm1d(768)
        self.dropout2 = nn.Dropout(0.4)
        self.output_layer = nn.Linear(768, num_classes)

    def forward(self, x):
        x = F.silu(self.dense1(x))
        x = self.bn1(x)
        x = self.dropout1(x)

        att = torch.sigmoid(self.attention(x))
        x_att = x * att

        metal_feat = self.metal_branch(x_att)
        punk_feat = self.punk_branch(x_att)
        blues_feat = self.blues_branch(x_att)

        x_concat = torch.cat([x_att, metal_feat, punk_feat, blues_feat], dim=1)

        x = F.silu(self.final_dense(x_concat))
        x = self.bn2(x)
        x = self.dropout2(x)

        out = torch.sigmoid(self.output_layer(x))
        return out


def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=22050, duration=30)
    
    features = {
        'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20),
        'spectral_contrast': librosa.feature.spectral_contrast(y=y, sr=sr),
        'spectral_flatness': librosa.feature.spectral_flatness(y=y),
        'zero_crossing_rate': librosa.feature.zero_crossing_rate(y=y),
        'chroma_cqt': librosa.feature.chroma_cqt(y=y, sr=sr),
        'tonnetz': librosa.feature.tonnetz(y=y, sr=sr),
        'tempogram': librosa.feature.tempogram(onset_envelope=librosa.onset.onset_strength(y=y, sr=sr)),
        'tempo': librosa.feature.rhythm.tempo(y=y, sr=sr)[0]
    }
    feature_vector = np.concatenate([
        np.mean(feat, axis=1) if isinstance(feat, np.ndarray) else [feat]
        for feat in features.values()
    ])
    return feature_vector

def load_mood_model(input_dim, num_classes):
    model = MoodClassifier(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load(MOOD_MODEL_PATH, map_location=device))
    model.eval()
    return model

def load_genre_model(input_dim, num_classes):
    model = MusicGenreModel(input_dim, num_classes).to(device)
    model.load_state_dict(torch.load(GENRE_MODEL_PATH, map_location=device))
    model.eval()
    return model

def predict(filepath):
    feat = extract_features(filepath).reshape(1, -1)
    
    # Load scalers and mlbs
    mood_scaler = joblib.load(MOOD_SCALER_PATH)
    mood_mlb = joblib.load(MOOD_MLB_PATH)
    genre_scaler = joblib.load(GENRE_SCALER_PATH)
    genre_mlb = joblib.load(GENRE_MLB_PATH)
    genre_thresholds = joblib.load(GENRE_THRESHOLDS_PATH)

    # Prepare mood prediction
    mood_X = mood_scaler.transform(feat)
    mood_model = load_mood_model(mood_X.shape[1], len(mood_mlb.classes_))
    mood_tensor = torch.tensor(mood_X, dtype=torch.float32).to(device)

    with torch.no_grad():
        mood_outputs = mood_model(mood_tensor)
        mood_probs = torch.softmax(mood_outputs, dim=1).cpu().numpy()
    mood_pred_idx = np.argmax(mood_probs, axis=1)[0]
    mood_pred = mood_mlb.classes_[mood_pred_idx]

    # Prepare genre prediction
    genre_X = genre_scaler.transform(feat)
    genre_model = load_genre_model(genre_X.shape[1], len(genre_mlb.classes_))
    genre_tensor = torch.tensor(genre_X, dtype=torch.float32).to(device)

    with torch.no_grad():
        genre_outputs = genre_model(genre_tensor)
        genre_probs = genre_outputs.cpu().numpy()[0]

    # Apply thresholds
    predicted_genres = [genre for genre, prob in zip(genre_mlb.classes_, genre_probs) 
                        if prob >= genre_thresholds.get(genre, 0.5)]

    return predicted_genres, mood_pred

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print(json.dumps({"error": "No audio file provided"}))
        sys.exit(1)

    audio_file = sys.argv[1]
    try:
        genres, mood = predict(audio_file)
        print(json.dumps({"genres": genres, "mood": mood}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
