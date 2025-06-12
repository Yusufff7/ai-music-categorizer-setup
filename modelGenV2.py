import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import MultiLabelBinarizer
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from config.genres import MAIN_GENRES, SUBGENRE_MAP

AUDIO_DIR = os.path.join('data', 'audio')
FEATURE_DIR = os.path.join('data', 'features')
os.makedirs(FEATURE_DIR, exist_ok=True)

def extract_features_cached(filepath):
    """Extract and cache features"""
    cache_path = os.path.join(FEATURE_DIR, filepath.replace('/', '__') + '.npy')
    
    if os.path.exists(cache_path):
        return np.load(cache_path)
    
    try:
        full_path = os.path.join(AUDIO_DIR, filepath)
        audio, sr = librosa.load(full_path, sr=22050, duration=30)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features = np.concatenate([np.mean(mfcc, axis=1), np.mean(chroma, axis=1)])
        np.save(cache_path, features)
        return features
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None  

def load_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv(os.path.join('data', 'raw_30s_cleantags.tsv'), sep='\t')
    
    df['genres'] = df['TAGS'].apply(
        lambda x: [t.replace('genre---', '') 
                  for t in x.split() if t.startswith('genre---')]
    )
    
    df['main_genres'] = df['genres'].apply(
        lambda tags: list({
            SUBGENRE_MAP.get(tag, tag) 
            for tag in tags 
            if SUBGENRE_MAP.get(tag, tag) in MAIN_GENRES
        })
    )

    df = df[df['main_genres'].map(len) > 0].reset_index(drop=True)
    
    return df


df = load_data()

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['main_genres'])

print("Extracting features...")
features = [extract_features_cached(path) for path in df['PATH']]
valid = [f is not None for f in features]
X = np.array([f for f in features if f is not None])
y = y[valid]

scaler = StandardScaler()
X = scaler.fit_transform(X)

y_strat = ['+'.join(genres) for genres in df['main_genres']]
y_strat = np.array(y_strat)[valid]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y_strat
)

model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.5),
    layers.Dense(y.shape[1], activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
model.save(os.path.join('data', 'genre_model.keras'))
print("Model trained and saved!")

joblib.dump(mlb, os.path.join('data', 'mlb.pkl'))
joblib.dump(scaler, os.path.join('data', 'scaler.pkl'))



