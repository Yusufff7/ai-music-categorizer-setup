import numpy as np
import torch
from tabulate import tabulate
from sklearn.metrics import classification_report

def diagnostic_report_torch(model, X_tensor, y_np, mlb, class_weights=None, thresholds=None):
    print("\n=== \U0001F31F Diagnostic Report (PyTorch) ===\n")

    # 1. Genre label counts in training set
    print("\U0001F4CA Training label counts per genre:")
    genre_counts = np.sum(y_np, axis=0)
    rows = [(genre, int(count)) for genre, count in zip(mlb.classes_, genre_counts)]
    print(tabulate(rows, headers=["Genre", "Count"]))

    print(f"\nTotal training samples: {len(X_tensor)}")

    # 2. Model architecture summary
    print("\n\U0001F9E0 Model architecture:")
    print(model)

    # 3. Final layer biases 
    try:
        found_bias = False
        for name, param in model.named_parameters():
            if "bias" in name and param.ndim == 1 and len(param) == len(mlb.classes_):
                print("\n\U0001F50E Final layer biases:")
                for genre, bias in zip(mlb.classes_, param.detach().cpu().numpy()):
                    print(f"  {genre:15s}: {bias:.4f}")
                found_bias = True
                break
        if not found_bias:
            print("\n Final layer biases not found or do not match expected dimensions.")
    except Exception as e:
        print(f"\n Error accessing final layer biases: {e}")

    # 4. Class weights
    if class_weights is not None:
        print("\n Class weights:")
        for genre, weight in zip(mlb.classes_, class_weights):
            print(f"  {genre:15s}: {weight:.2f}")
    else:
        print("\n No class weights provided.")

    # 5. Sample raw predictions
    print("\n\U0001F4C8 Sample raw predictions (first 3 samples):")
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor[:3]).cpu().numpy()

    for i, pred_row in enumerate(preds):
        print(f"\nSample {i+1}:")
        top_preds = sorted(zip(mlb.classes_, pred_row), key=lambda x: x[1], reverse=True)[:10]
        for genre, val in top_preds:
            print(f"  {genre:15s}: {val:.3f}")

    # 6. Classification report on small subset
    print("\n\U0001F522 Short classification report (first 100 samples):")
    y_true_small = y_np[:100]
    y_pred_small = model(X_tensor[:100]).cpu().numpy()

    if thresholds is not None:
        threshold_arr = np.array([thresholds.get(g, 0.5) for g in mlb.classes_])
        y_pred_bin = (y_pred_small > threshold_arr).astype(int)
    else:
        y_pred_bin = (y_pred_small > 0.5).astype(int)

    report = classification_report(y_true_small, y_pred_bin, target_names=mlb.classes_, zero_division=0, digits=3)
    print(report)

    print("\n=== End of Diagnostic Report ===")
