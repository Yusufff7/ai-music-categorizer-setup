import joblib

def edit_thresholds(file_path, updates):
    thresholds = joblib.load(file_path)
    print("Original thresholds:")
    print(thresholds)
    for k, v in updates.items():
        if k in thresholds:
            print(f"Updating {k}: {thresholds[k]} -> {v}")
            thresholds[k] = v
        else:
            print(f"Warning: {k} not found in thresholds")
    joblib.dump(thresholds, file_path)
    print(f"Updated thresholds saved to {file_path}")

if __name__ == "__main__":
    path = 'data/genre_thresholds.pkl'
    changes = {
        'pop': 0.37,
        'soundtrack': 0.26,
    }
    edit_thresholds(path, changes)
