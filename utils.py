import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from config.genres import MAIN_GENRES, SUBGENRE_MAP
from joblib import Parallel, delayed 
from datetime import datetime
from psutil import virtual_memory  
from tqdm import tqdm

def print_step(message, level=1):
    """Helper function for consistent step printing"""
    prefix = "  " * (level-1) + "↳" if level > 1 else ""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {prefix} {message}")

AUDIO_DIR = os.path.join('data', 'audio')
FEATURE_DIR = os.path.join('data', 'features')
os.makedirs(FEATURE_DIR, exist_ok=True)

def extract_features_cached(filepath):
    cache_path = os.path.join(FEATURE_DIR, filepath.replace('/', '__') + '.npy')
    if os.path.exists(cache_path):
        #print_step(f"Loading cached features for {filepath}", 2)
        return np.load(cache_path)

    try:
        print_step(f"Extracting features for {filepath}", 2)
        full_path = os.path.join(AUDIO_DIR, filepath)
 
        # Add file validation
        if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
            print_step(f"Invalid file: {filepath}", 2)
            return None
               
        print_step("Loading audio file...", 3)
        y, sr = librosa.load(full_path, sr=22050, duration=30)

        print_step("Extracting comprehensive features...", 3)
        features = {
            # Spectral/Timbre
            'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20),
            'spectral_contrast': librosa.feature.spectral_contrast(y=y, sr=sr),
            'spectral_flatness': librosa.feature.spectral_flatness(y=y),
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(y=y),
            
            # Harmonic
            'chroma_cqt': librosa.feature.chroma_cqt(y=y, sr=sr),
            'tonnetz': librosa.feature.tonnetz(y=y, sr=sr),
            
            # Rhythm
            'tempogram': librosa.feature.tempogram(onset_envelope=librosa.onset.onset_strength(y=y, sr=sr)),
            'tempo': librosa.beat.tempo(y=y, sr=sr)[0]
        }
        
        # Aggregate features
        feature_vector = np.concatenate([
            np.mean(feat, axis=1) if isinstance(feat, np.ndarray) 
            else [feat]  # For scalar values like tempo
            for feat in features.values()
        ])
        
        print_step(f"Caching features to {cache_path}", 3)
        np.save(cache_path, feature_vector)
        return feature_vector
        
    except Exception as e:
        print_step(f"Error processing {filepath}: {e}", 2)
        return None

def load_dataframe(tag_type):
    print_step(f"Loading dataframe for tag type: {tag_type}")
    
    print_step("Reading raw TSV file...", 2)
    with open(os.path.join('data', 'raw_30s_cleantags.tsv'), 'r', encoding='utf-8') as f:
        lines = [line.strip('\r\n') for line in f.readlines()]
    print_step(f"Found {len(lines)} entries in raw data", 2)
    
    print_step("Processing lines...", 2)
    data = []
    for i, line in enumerate(lines):
        if i % 1000 == 0:  # Progress indicator
            print_step(f"Processing line {i}/{len(lines)}", 3)
        parts = line.split('\t')
        metadata = parts[:5]
        tags = parts[5:]
        data.append(metadata + [tags])
    
    print_step("Creating DataFrame...", 2)
    df = pd.DataFrame(data, columns=['TRACK', 'ARTIST', 'ALBUM', 'PATH', 'DURATION', 'TAGS'])
    
    print_step("Converting duration to float...", 2)
    df['DURATION'] = pd.to_numeric(df['DURATION'], errors='coerce')
    
    print_step("Processing tags...", 2)
    def collect_tags(row):
        tags = []
        for tag in row['TAGS']:
            if isinstance(tag, list):
                tags.extend(tag)
            elif pd.notna(tag):
                tags.extend(str(tag).split())
        return [t.strip() for t in tags if t and t.strip()]
    
    df['TAGS'] = df.apply(collect_tags, axis=1)
    if tag_type == 'genre':
        print_step("Extracting genres...", 2)
        df['genres'] = df['TAGS'].apply(
            lambda tags: [t.replace('genre---', '') for t in tags 
                        if t.startswith('genre---')]
        )
        
        print_step("Inferring main genres...", 2)
        # Single-pass processing
        df['main_genres'], unmapped_results = zip(*df.apply(infer_main_genres, axis=1))
        all_unmapped = set().union(*[set(x) for x in unmapped_results if x])

        if all_unmapped:
            print_step(f"Found {len(all_unmapped)} unmapped subgenres:", 2)
            for tag in sorted(all_unmapped):
                print_step(f" - {tag}", 3)

        print_step("Filtering invalid tracks...", 2)
        df = df[df['main_genres'].map(len) > 0].reset_index(drop=True)
    elif tag_type == 'mood':
        print_step("Extracting mood tags...", 2)
        df['moods'] = df['TAGS'].apply(
            lambda tags: [t.replace('mood/theme---', '') for t in tags 
                        if t.startswith('mood/theme---')]
        )
        
        print_step("Filtering invalid tracks...", 2)
        
        # Count all mood occurrences
        all_moods = [m for sublist in df['moods'] for m in sublist]
        mood_counts = pd.Series(all_moods).value_counts()

        # Keep only moods with 100+ instances
        valid_moods = set(mood_counts[mood_counts >= 500].index)

        # Filter each track's moods to only valid moods
        df['moods'] = df['moods'].apply(lambda ms: [m for m in ms if m in valid_moods])

        # Drop rows with no valid moods left
        df = df[df['moods'].map(len) > 0].reset_index(drop=True)

        print(f"Filtered dataset to moods with >=500 instances, now {len(df)} tracks")
    else:
        raise ValueError(f"Unknown tag_type: {tag_type}")
    
    print_step(f"Final dataset contains {len(df)} valid tracks", 2)
    return df

def prepare_dataset(df, label_column, mlb=None, scaler=None):    
    print_step(f"Preparing dataset with label column: {label_column}")
    print_step("Analyzing label distribution...", 2)
    all_labels = [label for sublist in df[label_column] for label in sublist]
    label_counts = pd.Series(all_labels).value_counts()
    print_step("Label counts:\n" + str(label_counts), 3)

    print_step("Extracting features (parallel)...", 2)
    def batch_extract(paths):
        free_ram = virtual_memory().available / (1024**3)  # GB
        safe_jobs = max(1, min(6, int(free_ram / 1.5)))  # 1.5GB per job buffer
        
        return Parallel(n_jobs=safe_jobs,
                       verbose=10,
                       batch_size='auto',
                       max_nbytes='256M',
                       prefer="processes")(
            delayed(extract_features_cached)(path)
            for path in paths
        )
    
    # Process in batches of 5,000 tracks
    batch_size = 5000
    features = []
    batches = tqdm(
        [df['PATH'].iloc[i:i+batch_size] for i in range(0, len(df), batch_size)],
        desc="Extracting features",
        unit="batch",
        dynamic_ncols=True
    )
    
    for batch_paths in batches:
        features.extend(batch_extract(batch_paths))
        batches.set_postfix_str(f"Memory: {virtual_memory().percent}% used")
    
    print_step("Filtering valid features...", 2)
    valid = [f is not None for f in features]
    X = np.array([f for f in features if f is not None])
    print_step(f"Retained {len(X)}/{len(features)} valid feature sets", 2)
    
    # Uncomment for Cache Cleaning
    # print_step("Cleaning invalid cache entries...", 2)
    # valid_paths = set(df['PATH'].apply(lambda x: x.replace('/', '__')))  # Match cache naming
    # for f in os.listdir(FEATURE_DIR):
    #     if f.endswith('.npy') and not any(fp in f for fp in valid_paths):
    #         try:
    #             os.remove(os.path.join(FEATURE_DIR, f))
    #         except Exception as e:
    #             print_step(f"Couldn't remove {f}: {str(e)}", 3)
    
    # print_step(f"Cache cleanup complete ({FEATURE_DIR})", 2)

    print_step("Preparing labels...", 2)
    y_raw = df[label_column].values[valid]

    # Print label distribution after filtering
    print_step("Analyzing filtered label distribution...", 2)
    filtered_labels = [label for sublist in y_raw for label in sublist]
    filtered_counts = pd.Series(filtered_labels).value_counts()
    print_step("Filtered label counts:\n" + str(filtered_counts), 3)

    if mlb is None:
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(y_raw)
        print_step(f"Fitted new MultiLabelBinarizer with {len(mlb.classes_)} labels", 2)
    else:
        y = mlb.transform(y_raw)
        print_step("Used provided MultiLabelBinarizer", 2)

    # Scaling
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print_step("Fitted new StandardScaler", 2)
    else:
        X_scaled = scaler.transform(X)
        print_step("Used provided StandardScaler", 2)
    
    print_step("Dataset preparation complete", 2)
    return X_scaled, y, mlb, scaler, np.array(y_raw)

# Helper functions moved outside for clarity
def infer_main_genres(row):
    main_genres = set()
    unmapped_subgenres = set()
    
    for g in row['genres']:
        # Check if it's a main genre
        if g in MAIN_GENRES:
            main_genres.add(g)
            continue
            
        # Attempt to map subgenre
        mapped = SUBGENRE_MAP.get(g, None)
        if mapped in MAIN_GENRES:
            main_genres.add(mapped)
        else:
            unmapped_subgenres.add(g)  # Track but don't include
    
    return list(main_genres), unmapped_subgenres
    
def compute_balanced_weights(class_freqs, smoothing=0.15):
    inv_freq = {k: 1 / (v + 1e-6) for k, v in class_freqs.items()}
    max_inv = max(inv_freq.values())
    weights = {k: (v / max_inv) for k, v in inv_freq.items()}  # Normalize to [0, 1]
    weights = {k: smoothing + (1 - smoothing) * w for k, w in weights.items()}  # Smooth
    return weights


