"""
End-to-end Landslide Identification Pipeline.

Phase 1: MSAFusionNet ensemble (ConvNeXt-CBAM-FPN + SwinV2-Small) → 6-class classification
Phase 2: Bi-LSTM + Multi-Head Attention → temporal type + occurrence forecast

Usage:
    python pipeline/run_pipeline.py --mode train
    python pipeline/run_pipeline.py --mode predict --image /path/to/image.jpg --country India
    python pipeline/run_pipeline.py --mode evaluate
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from phase1_alexnet.predict import load_model, predict_single_image
from phase1_alexnet.ensemble_predict import load_ensemble, predict_ensemble
from phase1_alexnet.train import run_training

# Phase 2: Try LSTM first, fall back to HMM
LSTM_AVAILABLE = False
HMM_AVAILABLE = False

try:
    from phase2_lstm.temporal_predict import (
        load_model_and_encoders as load_lstm_model_and_encoders,
        classify_and_forecast as lstm_classify_and_forecast,
        format_phase2_output as lstm_format_output,
    )
    from phase2_lstm.temporal_train import train_temporal_model
    from phase2_lstm.data_preprocessing import (
        load_glc_data,
        encode_features,
        get_country_risk_profile,
        build_observation_sequences,
    )
    LSTM_AVAILABLE = True
except ImportError as e:
    print(f"LSTM Phase 2 not available: {e}")

try:
    from phase2_hmm.data_preprocessing import (
        load_glc_data as hmm_load_glc_data,
        encode_features as hmm_encode_features,
        build_observation_sequences as hmm_build_obs_seq,
        get_country_risk_profile as hmm_get_country_risk_profile,
    )
    from phase2_hmm.hmm_model import (
        build_hmm, load_hmm_model, save_hmm_model,
        load_encoder, save_encoder, train_hmm,
    )
    from phase2_hmm.hmm_predict import (
        classify_and_forecast as hmm_classify_and_forecast,
        format_phase2_output as hmm_format_output,
    )
    HMM_AVAILABLE = True
except ImportError:
    pass

# Use LSTM data_preprocessing if available, else HMM's
if not LSTM_AVAILABLE and HMM_AVAILABLE:
    load_glc_data = hmm_load_glc_data
    encode_features = hmm_encode_features
    get_country_risk_profile = hmm_get_country_risk_profile

from utils.plot_utils import plot_training_history

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class LandslideIdentificationPipeline:
    """
    Two-phase unified pipeline:
      Phase 1 — MSAFusionNet ensemble → 6-class landslide type classification
      Phase 2 — Bi-LSTM + Attention → temporal forecasting (type, probability, future)
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Phase 1: Load ensemble ────────────────────────────────────────
        print("Loading Phase 1 ensemble (ConvNeXt-CBAM-FPN + SwinV2-Small) ...")
        self.convnext, self.swinv2, _ = load_ensemble(self.device)

        # ── Phase 2: Load temporal model ──────────────────────────────────
        self.phase2_mode = None
        self.risk_profile = None
        self.df_encoded = None

        if config.ACTIVE_PHASE2 == "lstm" and LSTM_AVAILABLE and os.path.exists(config.LSTM_MODEL_PATH):
            print("Loading Phase 2: Bi-LSTM + Attention ...")
            self.lstm_model, self.type_encoder, self.trigger_encoder, _ = load_lstm_model_and_encoders(self.device)
            df = load_glc_data()
            self.df_encoded, _, _ = encode_features(df)
            self.risk_profile = get_country_risk_profile(self.df_encoded)
            self.phase2_mode = "lstm"
            print(f"Phase 2 ready (LSTM). Risk profiles for {len(self.risk_profile)} countries.\n")

        elif HMM_AVAILABLE and os.path.exists(config.HMM_MODEL_PATH):
            print("Loading Phase 2: HMM (legacy) ...")
            self.hmm = load_hmm_model()
            self.type_encoder = load_encoder()
            df = hmm_load_glc_data()
            df_enc, _, _ = hmm_encode_features(df)
            self.risk_profile = hmm_get_country_risk_profile(df_enc)
            self.phase2_mode = "hmm"
            print(f"Phase 2 ready (HMM). Risk profiles for {len(self.risk_profile)} countries.\n")

        else:
            print("Phase 2 not available (no trained model found).\n")

        print("Pipeline ready.\n")

    def predict(
        self,
        image_path: str,
        country: str = None,
        n_forecast_steps: int = 3,
        obs_sequence: np.ndarray = None,
        location: str = None,
        **kwargs,
    ) -> dict:
        """
        Run end-to-end prediction for one image.

        Phase 1 → 6-class classification
        Phase 2 → temporal forecasting (if landslide detected)
        """
        # ── Phase 1 ──────────────────────────────────────────────────────
        phase1_result = predict_ensemble(
            image_path, self.convnext, self.swinv2, self.device
        )

        effective_country = country or location
        is_landslide = phase1_result.get("is_landslide", phase1_result["label"] != "non_landslide")
        landslide_type = phase1_result.get("landslide_type", phase1_result["label"])

        result = {
            "image_path": image_path,
            "phase1": phase1_result,
            "phase2": None,
            "final_verdict": "",
        }

        # ── Phase 2 (if landslide detected) ──────────────────────────────
        if is_landslide and self.phase2_mode:
            if self.phase2_mode == "lstm":
                p2 = lstm_classify_and_forecast(
                    self.lstm_model, self.type_encoder, self.risk_profile,
                    country=effective_country,
                    obs_sequence=obs_sequence,
                    n_forecast_steps=n_forecast_steps,
                    trigger_encoder=self.trigger_encoder,
                    df_encoded=self.df_encoded,
                )
            else:
                p2 = hmm_classify_and_forecast(
                    self.hmm, self.type_encoder, self.risk_profile,
                    country=effective_country,
                    obs_sequence=obs_sequence,
                    n_forecast_steps=n_forecast_steps,
                )

            p2_flat = {
                "landslide_type":         landslide_type,
                "phase1_type":            landslide_type,
                "hmm_type":               p2["current_type"],  # Keep key name for compat
                "temporal_model_type":    p2["current_type"],
                "occurrence_probability": p2["occurrence_probability"],
                "hidden_state":           p2.get("hidden_state", 0),
                "peak_risk_month":        p2["peak_risk_month"],
                "future_forecast":        p2["future_forecast"],
                "risk_stats":             p2["risk_stats"],
                "country":                p2["country"],
                "phase2_model":           self.phase2_mode,
                "next_event_forecast": [
                    {
                        "most_likely_state": f["landslide_type"],
                        "probability":       f["probability"],
                        "step":              f["step"],
                    }
                    for f in p2["future_forecast"]
                ],
            }
            # Add attention weights if LSTM
            if "attention_weights" in p2:
                p2_flat["attention_weights"] = p2["attention_weights"]

            result["phase2"] = p2_flat
            prob_pct = int(p2["occurrence_probability"] * 100)
            conf_pct = int(phase1_result["confidence"] * 100)
            result["final_verdict"] = (
                f"LANDSLIDE DETECTED — "
                f"Type: {landslide_type} (confidence: {conf_pct}%) "
                f"| Occurrence probability: {prob_pct}% "
                f"| Model: {self.phase2_mode.upper()}"
            )
        elif is_landslide:
            conf_pct = int(phase1_result["confidence"] * 100)
            result["final_verdict"] = (
                f"LANDSLIDE DETECTED — "
                f"Type: {landslide_type} (confidence: {conf_pct}%) "
                f"| Phase 2 offline"
            )
        else:
            conf_pct = int(phase1_result["confidence"] * 100)
            result["final_verdict"] = (
                f"NO LANDSLIDE DETECTED "
                f"(confidence: {conf_pct}%)"
            )

        return result

    def batch_predict(self, image_dir: str, output_csv: str = None,
                      country: str = None) -> pd.DataFrame:
        """Run prediction on all images in a directory."""
        image_paths = [
            str(p) for p in sorted(Path(image_dir).iterdir())
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        print(f"Found {len(image_paths)} images in {image_dir}")

        rows = []
        for img_path in tqdm(image_paths, desc="Batch predict"):
            try:
                res = self.predict(img_path, country=country)
                row = {
                    "image_path":            res["image_path"],
                    "phase1_label":          res["phase1"]["label"],
                    "phase1_confidence":     res["phase1"]["confidence"],
                    "phase1_is_landslide":   res["phase1"].get("is_landslide", False),
                    "phase1_landslide_prob": res["phase1"].get("landslide_prob", 0),
                    "phase2_type":           res["phase2"]["landslide_type"] if res["phase2"] else None,
                    "phase2_probability":    res["phase2"]["occurrence_probability"] if res["phase2"] else None,
                    "phase2_peak_month":     res["phase2"]["peak_risk_month"] if res["phase2"] else None,
                    "phase2_model":          res["phase2"]["phase2_model"] if res["phase2"] else None,
                    "verdict":               res["final_verdict"],
                }
            except Exception as e:
                row = {"image_path": img_path, "error": str(e)}
            rows.append(row)

        df = pd.DataFrame(rows)
        if output_csv:
            os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")
        return df

    def run_full_evaluation(self) -> dict:
        """Evaluate both phases and return a combined report."""
        report = {}

        # ── Phase 1 ──────────────────────────────────────────────────────
        from phase1_alexnet.evaluate import run_evaluation

        print("=== Phase 1 Evaluation (ConvNeXt-CBAM-FPN) ===")
        p1_convnext = run_evaluation(model_name="convnext_cbam_fpn")
        report["phase1_convnext"] = {
            "accuracy": p1_convnext["accuracy"],
            "f1": p1_convnext["f1"],
            "roc_auc": p1_convnext.get("roc_auc"),
        }

        print("\n=== Phase 1 Evaluation (SwinV2-Small) ===")
        p1_swinv2 = run_evaluation(model_name="swinv2_s")
        report["phase1_swinv2"] = {
            "accuracy": p1_swinv2["accuracy"],
            "f1": p1_swinv2["f1"],
            "roc_auc": p1_swinv2.get("roc_auc"),
        }

        # ── Phase 2 ──────────────────────────────────────────────────────
        if self.phase2_mode == "lstm":
            print("\n=== Phase 2 Evaluation (Bi-LSTM + Attention) ===")
            # Evaluate on test sequences
            from phase2_lstm.data_preprocessing import build_sequences
            df = load_glc_data()
            df_enc, type_enc, trig_enc = encode_features(df)
            sequences = build_sequences(df_enc, type_enc, trig_enc)

            correct = 0
            total = 0
            for features, types, country in sequences:
                if len(features) < 3:
                    continue
                input_feats = torch.tensor(features[:-1], dtype=torch.float32).unsqueeze(0).to(self.device)
                lengths = torch.tensor([len(features) - 1], dtype=torch.long).to(self.device)
                target = types[-1]

                type_probs, _, _, _ = self.lstm_model.predict_step(input_feats, lengths)
                pred = type_probs[0].argmax().item()
                if pred == target:
                    correct += 1
                total += 1

            p2_acc = correct / total if total > 0 else 0
            print(f"LSTM type prediction accuracy: {p2_acc:.4f} ({correct}/{total})")
            report["phase2"] = {
                "model": "Bi-LSTM + Attention",
                "type_accuracy": p2_acc,
                "n_sequences": total,
            }

        elif self.phase2_mode == "hmm" and HMM_AVAILABLE:
            print("\n=== Phase 2 Evaluation (HMM — legacy) ===")
            df = hmm_load_glc_data()
            df_enc, _, _ = hmm_encode_features(df)
            seqs, lengths, _ = hmm_build_obs_seq(df_enc)
            X = np.concatenate(seqs).astype(int).reshape(-1, 1)
            ll = self.hmm.score(X, lengths)
            report["phase2"] = {
                "model": "HMM",
                "log_likelihood": ll,
                "per_obs_log_ll": ll / sum(lengths),
            }

        print("\n=== Full Report ===")
        print(json.dumps(report, indent=2, default=str))
        return report


# ─── Training ─────────────────────────────────────────────────────────────────

def train_phase1():
    """Train both Phase 1 models."""
    print("=" * 60)
    print("Training MSAFusionNet (ConvNeXt-Base + CBAM + FPN)")
    print("=" * 60)
    model1, history1 = run_training(model_name="convnext_cbam_fpn")
    plot_training_history(
        history1,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_convnext.png"),
    )

    print("\n" + "=" * 60)
    print("Training SwinV2-Small (ensemble partner)")
    print("=" * 60)
    model2, history2 = run_training(model_name="swinv2_s")
    plot_training_history(
        history2,
        save_path=os.path.join(config.PLOTS_DIR, "training_history_swinv2.png"),
    )


def train_phase2():
    """Train Phase 2 model (LSTM or HMM based on config)."""
    if config.ACTIVE_PHASE2 == "lstm" and LSTM_AVAILABLE:
        print("\n" + "=" * 60)
        print("Training Phase 2: Bi-LSTM + Multi-Head Attention")
        print("=" * 60)
        model, history, _, _ = train_temporal_model()
        plot_training_history(
            history,
            save_path=os.path.join(config.PLOTS_DIR, "training_history_lstm.png"),
        )
    elif HMM_AVAILABLE:
        import joblib
        print("\n=== Training Phase 2: HMM (legacy) ===")
        df = hmm_load_glc_data()
        df_enc, type_enc, trig_enc = hmm_encode_features(df)
        use_combined = config.HMM_USE_COMBINED_OBS
        seqs, lengths, _ = hmm_build_obs_seq(df_enc, use_combined=use_combined)
        n_types = len(type_enc.classes_)
        n_triggers = int(df_enc["trigger_code"].max() + 1)
        n_symbols = n_types * n_triggers if use_combined else n_types
        model = build_hmm(n_symbols=int(n_symbols))
        model = train_hmm(model, seqs, lengths)
        save_hmm_model(model)
        save_encoder(type_enc)
        joblib.dump(trig_enc, config.HMM_TRIGGER_ENCODER_PATH)
        joblib.dump({"use_combined": use_combined, "n_types": n_types, "n_triggers": n_triggers},
                    os.path.join(config.CHECKPOINTS_DIR, "hmm_meta.pkl"))
    else:
        print("No Phase 2 model available for training.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Landslide Identification Pipeline (MSAFusionNet + Bi-LSTM)"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "evaluate"],
        required=True,
    )
    parser.add_argument("--image", help="Path to image (for --mode predict)")
    parser.add_argument("--country", default=None, help="Country for Phase 2")
    parser.add_argument("--forecast", type=int, default=3, help="Forecast steps")
    args = parser.parse_args()

    if args.mode == "train":
        train_phase1()
        train_phase2()
        print("\nAll models trained successfully.")

    elif args.mode == "predict":
        if not args.image:
            parser.error("--image is required for --mode predict")
        pipeline = LandslideIdentificationPipeline()
        result = pipeline.predict(
            args.image,
            country=args.country,
            n_forecast_steps=args.forecast,
        )
        print("\n=== Prediction Result ===")
        p1 = result["phase1"]
        print(f"Phase 1 — {p1['label'].upper()} (confidence: {p1['confidence'] * 100:.1f}%)")
        print(f"  Per-class probabilities:")
        for cls, prob in p1["probabilities"].items():
            marker = " <--" if cls == p1["label"] else ""
            print(f"    {cls:>20s}: {prob * 100:.2f}%{marker}")
        if result["phase2"]:
            p2 = result["phase2"]
            print(f"Phase 2 ({p2['phase2_model'].upper()}) — "
                  f"Type: {p2['temporal_model_type']}")
            print(f"  Occurrence probability: {p2['occurrence_probability'] * 100:.1f}%")
            print(f"  Peak risk month: {p2['peak_risk_month']}")
            print(f"  Future forecast:")
            for f in p2["future_forecast"]:
                print(f"    Step {f['step']}: {f['landslide_type']} "
                      f"(prob={f['probability'] * 100:.1f}%)")
        print(f"\n{result['final_verdict']}")

    elif args.mode == "evaluate":
        pipeline = LandslideIdentificationPipeline()
        pipeline.run_full_evaluation()


if __name__ == "__main__":
    main()
