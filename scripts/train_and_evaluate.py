"""
Test and benchmark script for Phase 2:
1. Compares 5 candidate models on seeded data.
2. Evaluates Before vs After Improvement pipeline.
3. Tunes and evaluates Multimodal Fusion.
4. Saves models to disk and records evaluation results to SQLite.
5. Prints comprehensive biometric metrics.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from core.db import get_session, User, EvaluationResult, ModelRecord
from core.models import (
    load_dataset,
    compare_models,
    run_improvement_benchmark,
    train_and_save_user_models,
    load_user_fusion,
    load_user_pipeline,
)
from core.metrics import evaluate_predictions


def main():
    print("=" * 65)
    print("BIOAUTH LAB - ML CORE BENCHMARK & EVALUATION")
    print("=" * 65)

    session = get_session()
    user = session.query(User).filter_by(username="user_01").first()
    if not user:
        print("[!] User user_01 not found. Please run scripts/seed.py first.")
        session.close()
        return

    user_id = user.id
    print(f"[*] Evaluating for User: {user.username} (ID: {user_id})\n")

    # -------------------------------------------------------------
    # 1. Compare 5 Candidate Models across Modalities
    # -------------------------------------------------------------
    modalities = ["keystroke", "mouse", "touch"]

    for mod in modalities:
        print(f"\n--- [1] CANDIDATE MODEL COMPARISON: {mod.upper()} ---")
        X_df, y, feat_names = load_dataset(user_id=user_id, modality=mod)
        print(f"Dataset shape: {X_df.shape[0]} samples, {X_df.shape[1]} features (Genuine: {sum(y==1)}, Impostor: {sum(y==0)})")

        comp = compare_models(X_df.values, y, n_splits=5)
        print(f"Best Model by CV EER: '{comp['best_model']}' (EER = {comp['best_eer']:.4f})")

        # Tabulate comparison
        table_rows = []
        for m_name, m_res in comp["results"].items():
            s = m_res["summary_metrics"]
            table_rows.append({
                "Model": m_name,
                "Mean EER": f"{m_res['mean_eer']:.4f}",
                "AUC": f"{s['auc']:.4f}",
                "Accuracy": f"{s['accuracy']:.4f}",
                "Precision": f"{s['precision']:.4f}",
                "Recall": f"{s['recall']:.4f}",
                "F1 Score": f"{s['f1']:.4f}",
            })
        df_comp = pd.DataFrame(table_rows)
        print(df_comp.to_string(index=False))

    # -------------------------------------------------------------
    # 2. Before vs After Improvement Pipeline
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("--- [2] BEFORE vs AFTER IMPROVEMENT PIPELINE ---")
    print("=" * 65)

    for mod in modalities:
        bench = run_improvement_benchmark(user_id=user_id, modality=mod)
        b = bench["before"]
        a = bench["after"]
        print(f"\nModality: {mod.upper()}")
        print(f"  * BEFORE Pipeline (Default Baseline):")
        print(f"      EER: {b['eer']:.4f} | AUC: {b['auc']:.4f} | FAR: {b['far']:.4f} | FRR: {b['frr']:.4f} | Acc: {b['accuracy']:.4f} | F1: {b['f1']:.4f}")
        print(f"  * AFTER Pipeline (Outlier Removal + Scaler + SelectKBest + GridSearch):")
        print(f"      EER: {a['eer']:.4f} | AUC: {a['auc']:.4f} | FAR: {a['far']:.4f} | FRR: {a['frr']:.4f} | Acc: {a['accuracy']:.4f} | F1: {a['f1']:.4f}")
        print(f"  -> Delta EER: {bench['eer_improvement']:+.4f} (Lower is better) | Delta AUC: {bench['auc_improvement']:+.4f}")

    # -------------------------------------------------------------
    # 3. Train, Save Models and Multimodal Score Fusion
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("--- [3] TRAINING, PERSISTENCE & SCORE-LEVEL FUSION ---")
    print("=" * 65)

    train_summary = train_and_save_user_models(user_id=user_id)
    print(f"Trained & Saved Pipelines for modalities: {train_summary['modalities']}")
    fusion_info = train_summary.get("fusion", {})
    weights = fusion_info.get("weights", [])
    fusion_metrics = fusion_info.get("metrics", {})

    print(f"\nMultimodal Fusion Weights (Keystroke, Mouse, Touch): {weights}")
    print(f"Fusion Verification Metrics:")
    print(f"  EER:        {fusion_metrics.get('eer', 'N/A')}")
    print(f"  AUC:        {fusion_metrics.get('auc', 'N/A')}")
    print(f"  Accuracy:   {fusion_metrics.get('accuracy', 'N/A')}")
    print(f"  Precision:  {fusion_metrics.get('precision', 'N/A')}")
    print(f"  Recall:     {fusion_metrics.get('recall', 'N/A')}")
    print(f"  F1 Score:   {fusion_metrics.get('f1', 'N/A')}")
    print(f"  FAR / FRR:  {fusion_metrics.get('far', 'N/A')} / {fusion_metrics.get('frr', 'N/A')}")
    print(f"  Confusion Matrix: {fusion_metrics.get('confusion_matrix', 'N/A')}")

    # -------------------------------------------------------------
    # 4. Also train models for a few other users to ensure DB is full
    # -------------------------------------------------------------
    print("\n[*] Training and recording benchmark for additional users...")
    all_users = session.query(User).all()
    for u in all_users[1:5]:  # users 2, 3, 4, 5
        train_and_save_user_models(u.id)
        for m in modalities:
            run_improvement_benchmark(u.id, m)

    # -------------------------------------------------------------
    # 5. Database Verification
    # -------------------------------------------------------------
    model_count = session.query(ModelRecord).count()
    result_count = session.query(EvaluationResult).count()
    print(f"\n[+] Total Saved Model Records in DB: {model_count}")
    print(f"[+] Total Saved Evaluation Results in DB: {result_count}")

    # Test loading serialized models and running inference
    kp = load_user_pipeline(user_id, "keystroke")
    mp = load_user_pipeline(user_id, "mouse")
    tp = load_user_pipeline(user_id, "touch")
    fus = load_user_fusion(user_id)
    assert kp is not None, "Failed to load saved keystroke pipeline!"
    assert mp is not None, "Failed to load saved mouse pipeline!"
    assert tp is not None, "Failed to load saved touch pipeline!"
    assert fus is not None, "Failed to load saved fusion model!"

    # Simulate inference for each modality with its matching sample
    X_k, _, _ = load_dataset(user_id=user_id, modality="keystroke")
    X_m, _, _ = load_dataset(user_id=user_id, modality="mouse")
    X_t, _, _ = load_dataset(user_id=user_id, modality="touch")

    score_k = float(kp.predict_proba(X_k.values[:1])[0, 1])
    score_m = float(mp.predict_proba(X_m.values[:1])[0, 1])
    score_t = float(tp.predict_proba(X_t.values[:1])[0, 1])
    f_score, contribs = fus.fuse(score_k, score_m, score_t)
    print(f"\n[Test Inference for User 01]")
    print(f"  Single Sub-scores -> Keystroke: {score_k:.3f}, Mouse: {score_m:.3f}, Touch: {score_t:.3f}")
    print(f"  Fused Genuine Score: {f_score:.3f}")
    print(f"  Modality Contributions: {contribs}")

    session.close()
    print("\n[+] Phase 2 ML Core successfully verified and saved!")


if __name__ == "__main__":
    main()
