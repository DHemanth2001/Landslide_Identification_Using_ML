"""
End-to-end Landslide Identification Pipeline.
Chains Phase 1 (AlexNet image classification) → Phase 2 (HMM type + future forecast).

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
from phase1_alexnet.predict import load_alexnet_model, predict_single_image
from phase1_alexnet.ensemble_predict import load_ensemble, predict_ensemble
from phase1_alexnet.train import run_training
try:
    from phase2_hmm.data_preprocessing import (
        load_glc_data,
        encode_features,
        build_observation_sequences,
        get_country_risk_profile,
    )
    from phase2_hmm.hmm_model import (
        build_hmm,
        load_encoder,
        load_hmm_model,
        save_encoder,
        save_hmm_model,
        train_hmm,
    )
    from phase2_hmm.hmm_predict import classify_and_forecast, format_phase2_output
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
from utils.plot_utils import plot_training_history

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class LandslideIdentificationPipeline:
    """
    Two-phase pipeline:
      Phase 1 — AlexNet determines if an image contains a landslide.
      Phase 2 — HMM classifies the type, occurrence probability, and future forecast.
    """

    def __init__(
        self,
        alexnet_checkpoint: str = None,
        hmm_model_path: str = None,
        hmm_encoder_path: str = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Phase 1 ensemble (EfficientNet-B3 + ViT-B/16)
        print("Loading Phase 1 ensemble (EfficientNet-B3 + ViT-B/16) ...")
        self.effnet, self.vit_model, _ = load_ensemble(self.device)

        # Load Phase 2 model + risk profile
        if HMM_AVAILABLE:
            hmm_path = hmm_model_path or config.HMM_MODEL_PATH
            enc_path = hmm_encoder_path or config.HMM_ENCODER_PATH
            print(f"Loading HMM from {hmm_path} ...")
            self.hmm = load_hmm_model(hmm_path)
            self.type_encoder = load_encoder(enc_path)

            # Build risk profile from NASA GLC for per-country stats
            print("Loading NASA GLC risk profiles ...")
            df = load_glc_data()
            df_enc, _, _ = encode_features(df)
            self.risk_profile = get_country_risk_profile(df_enc)
            print(f"Risk profiles loaded for {len(self.risk_profile)} countries.\n")
        else:
            print("HMM not available, skipping Phase 2 loading.\n")

        print("Pipeline ready.\n")

    def predict(
        self,
        image_path: str,
        country: str = None,
        n_forecast_steps: int = 3,
        obs_sequence: np.ndarray = None,
        # Legacy args kept for notebook compatibility
        disaster_description: str = None,
        location: str = None,
        duration: int = None,
        year: int = None,
    ) -> dict:
        """
        Run end-to-end prediction for one image.

        Args:
            image_path:       Path to the input image.
            country:          Country name for Phase 2 risk profile lookup.
            n_forecast_steps: Number of future time steps to forecast.
            obs_sequence:     Optional pre-built observation array for HMM.

        Returns:
            Unified result dict.
        """
        # ── Phase 1: Ensemble classification ─────────────────────────────────
        phase1_result = predict_ensemble(image_path, self.effnet, self.vit_model, self.device)

        # Support legacy 'location' arg as country
        effective_country = country or location

        result = {
            "image_path": image_path,
            "phase1": phase1_result,
            "phase2": None,
            "final_verdict": "",
        }

        # ── Phase 2: Type + probability (only if landslide detected) ─────────
        if (
            phase1_result["label"] == "landslide"
            and phase1_result["confidence"] >= config.PHASE1_THRESHOLD
        ):
            if HMM_AVAILABLE:
                p2 = classify_and_forecast(
                    self.hmm,
                    self.type_encoder,
                    self.risk_profile,
                    country=effective_country,
                    obs_sequence=obs_sequence,
                    n_forecast_steps=n_forecast_steps,
                )
                # Flatten for legacy consumers
                p2_flat = {
                    "landslide_type":        p2["current_type"],
                    "occurrence_probability": p2["occurrence_probability"],
                    "hidden_state":          p2["hidden_state"],
                    "peak_risk_month":       p2["peak_risk_month"],
                    "future_forecast":       p2["future_forecast"],
                    "risk_stats":            p2["risk_stats"],
                    "country":               p2["country"],
                    # Legacy key for notebook 05 compatibility
                    "next_event_forecast": [
                        {
                            "most_likely_state": f["landslide_type"],
                            "probability":       f["probability"],
                            "step":              f["step"],
                        }
                        for f in p2["future_forecast"]
                    ],
                }
                result["phase2"] = p2_flat
                prob_pct = int(p2["occurrence_probability"] * 100)
                result["final_verdict"] = (
                    f"LANDSLIDE DETECTED — "
                    f"Type: {p2['current_type']} "
                    f"(occurrence probability: {prob_pct}%)"
                )
            else:
                result["final_verdict"] = (
                    f"LANDSLIDE DETECTED — "
                    f"Phase 2 HMM offline. Confidence: {int(phase1_result['confidence']*100)}%"
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
        """
        Run prediction on all images in a directory.

        Args:
            image_dir:  Directory containing image files.
            output_csv: If provided, save results to this CSV path.
            country:    Country context for Phase 2.

        Returns:
            pd.DataFrame with one row per image.
        """
        image_paths = [
            str(p)
            for p in sorted(Path(image_dir).iterdir())
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        print(f"Found {len(image_paths)} images in {image_dir}")

        rows = []
        for img_path in tqdm(image_paths, desc="Batch predict"):
            try:
                res = self.predict(img_path, country=country)
                row = {
                    "image_path":           res["image_path"],
                    "phase1_label":         res["phase1"]["label"],
                    "phase1_confidence":    res["phase1"]["confidence"],
                    "phase1_landslide_prob": res["phase1"]["probabilities"]["landslide"],
                    "phase2_type":          res["phase2"]["landslide_type"] if res["phase2"] else None,
                    "phase2_probability":   res["phase2"]["occurrence_probability"] if res["phase2"] else None,
                    "phase2_peak_month":    res["phase2"]["peak_risk_month"] if res["phase2"] else None,
                    "verdict":              res["final_verdict"],
                }
            except Exception as e:
                row = {"image_path": img_path, "error": str(e)}
            rows.append(row)

        df = pd.DataFrame(rows)
        if output_csv:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")

        return df

    def run_full_evaluation(
        self, test_image_dir: str = None
    ) -> dict:
        """
        Evaluate both phases and return a combined report.

        Returns:
            dict with phase1 and phase2 evaluation summaries.
        """
        report = {}

        # ── Phase 1 evaluation ────────────────────────────────────────────────
        from phase1_alexnet.evaluate import run_evaluation
        print("=== Phase 1 Evaluation ===")
        phase1_results = run_evaluation()
        report["phase1"] = {
            "accuracy":  phase1_results["accuracy"],
            "precision": phase1_results["precision"],
            "recall":    phase1_results["recall"],
            "f1":        phase1_results["f1"],
            "roc_auc":   phase1_results.get("roc_auc"),
        }

        # ── Phase 2 evaluation ────────────────────────────────────────────────
        print("\n=== Phase 2 Evaluation ===")
        df = load_glc_data()
        df_enc, type_enc, _ = encode_features(df)
        seqs, lengths, labels = build_observation_sequences(df_enc)

        X = np.concatenate(seqs).astype(int).reshape(-1, 1)
        ll = self.hmm.score(X, lengths)
        print(f"HMM log-likelihood on all {sum(lengths)} observations: {ll:.4f}")
        print(f"Per-observation log-likelihood: {ll/sum(lengths):.4f}")
        report["phase2"] = {
            "log_likelihood":     ll,
            "per_obs_log_ll":     ll / sum(lengths),
            "n_records":          sum(lengths),
            "n_sequences":        len(seqs),
            "n_countries":        len(labels),
        }

        print("\n=== Full Evaluation Report ===")
        print(json.dumps(report, indent=2, default=str))
        return report


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_phase1(pretrained: bool = False, model_name: str = None) -> None:
    """Train Phase 1 model and save best checkpoint."""
    if model_name is None:
        model_name = config.ACTIVE_MODEL
    print(f"=== Training Phase 1: {model_name} ===")
    model, history = run_training(pretrained=pretrained, model_name=model_name)
    plot_training_history(
        history,
        save_path=os.path.join(config.PLOTS_DIR, "training_history.png"),
    )


def train_phase2() -> None:
    """Train HMM on NASA GLC records and save model + encoder."""
    import joblib
    print("\n=== Training Phase 2: HMM (NASA GLC) ===")
    df = load_glc_data()
    df_enc, type_enc, trig_enc = encode_features(df)
    use_combined = config.HMM_USE_COMBINED_OBS
    seqs, lengths, _ = build_observation_sequences(df_enc, use_combined=use_combined)

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
    print("Phase 2 training complete.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Landslide Identification Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "evaluate"],
        required=True,
        help="train: train both models; predict: run on a single image; evaluate: full evaluation",
    )
    parser.add_argument("--image", help="Path to image (required for --mode predict)")
    parser.add_argument("--country", default=None, help="Country name for HMM risk profile")
    parser.add_argument("--forecast", type=int, default=3, help="Number of future steps to forecast")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained AlexNet weights")
    args = parser.parse_args()

    if args.mode == "train":
        train_phase1(pretrained=args.pretrained, model_name=config.ACTIVE_MODEL)
        train_phase2()
        print("\nBoth models trained successfully.")

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
        print(f"Phase 1 — {p1['label'].upper()} (confidence: {p1['confidence']*100:.1f}%)")
        if result["phase2"]:
            p2 = result["phase2"]
            print(f"Phase 2 — Type: {p2['landslide_type']}")
            print(f"         Occurrence probability: {p2['occurrence_probability']*100:.1f}%")
            print(f"         Peak risk month: {p2['peak_risk_month']}")
            print(f"         Future forecast:")
            for f in p2["future_forecast"]:
                print(f"           Step {f['step']}: {f['landslide_type']} (prob={f['probability']*100:.1f}%)")
        print(f"\n{result['final_verdict']}")

    elif args.mode == "evaluate":
        pipeline = LandslideIdentificationPipeline()
        pipeline.run_full_evaluation()


if __name__ == "__main__":
    main()
