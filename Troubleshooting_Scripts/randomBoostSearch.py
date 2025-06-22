import copy
import numpy as np
from model_trainer import train_model
def run_random_boost_search(X, y, mlb, scaler, trials=30):
    genres = list(mlb.classes_)
    base_boosts = {
        'punk': 4.0, 'country': 4.0, 'industrial': 3.5,
        'blues': 3.0, 'reggae': 2.8, 'metal': 3.0,
        'jazz': 2.5, 'lounge': 2.3, 'world': 2.3, 'rnb': 2.0,
        'folk': 2.0, 'hiphop': 2.0,
        'classical': 2.0, 'rock': 2.0,
        'electronic': 0.4, 'pop': 1.8, 'soundtrack': 2.5,
        'alternative': 3.5
    }
    boost_ranges = {g: (0.3, 6.0) for g in genres}

    def sample_random_boosts():
        sampled = {}
        for genre in genres:
            base = base_boosts.get(genre, 1.0)
            low = max(0.3, base * 0.5)
            high = min(6.0, base * 1.5)
            sampled[genre] = float(np.random.uniform(low, high))
        return sampled

    best_macro_f1 = 0
    best_boosts = copy.deepcopy(base_boosts)

    for i in range(trials):
        boosts = sample_random_boosts()
        print(f"Trial {i+1}/{trials} boosts:")
        for g in sorted(boosts):
            print(f"  {g:12s}: {boosts[g]:.3f}")

        metrics = train_model(
            X, y, mlb,
            manual_boosts=boosts,
            scaler=scaler,
            epochs=25,
            verbose=False
        )

        macro_f1 = metrics['val_macro_f1']
        print(f"  Validation Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_boosts = boosts
            print("  --> New best boosts!")

    print("\nBest manual boosts found:")
    for g in sorted(best_boosts):
        print(f"{g:12s}: {best_boosts[g]:.3f}")
    print(f"Best validation macro F1: {best_macro_f1:.4f}")

    return best_boosts
