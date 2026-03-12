"""
Hidden Markov Model for landslide type classification.
Uses hmmlearn CategoricalHMM trained on NASA GLC temporal sequences.

Hidden states capture latent "landslide regimes" (e.g., rainfall-driven shallow slides,
seismic rockfall zone, etc.). Each state has an emission distribution over the 7 types.
"""

import os, sys, warnings
import joblib
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from hmmlearn import hmm as hmmlearn_hmm
    CategoricalHMM = getattr(hmmlearn_hmm, "CategoricalHMM", None) or hmmlearn_hmm.MultinomialHMM
except ImportError:
    raise ImportError("Install hmmlearn: pip install hmmlearn")


def build_hmm(n_components=None, n_symbols=None, n_iter=None):
    if n_components is None:
        n_components = config.HMM_N_COMPONENTS
    if n_iter is None:
        n_iter = config.HMM_N_ITER

    model = CategoricalHMM(
        n_components=n_components,
        n_iter=n_iter,
        tol=1e-4,
        algorithm="viterbi",
        random_state=config.RANDOM_SEED,
        verbose=False,
    )
    if n_symbols is not None:
        model.n_features = n_symbols
    print(f"HMM built: {n_components} hidden states, {n_symbols} observation symbols")
    return model


def train_hmm(model, obs_sequences, lengths):
    """Train HMM with Baum-Welch on all country sequences."""
    X = np.concatenate(obs_sequences).astype(int).reshape(-1, 1)
    print(f"Training on {X.shape[0]} observations across {len(lengths)} sequences...")
    model.fit(X, lengths)
    ll = model.score(X, lengths)
    print(f"Training complete. Log-likelihood: {ll:.4f}")
    return model


def save_hmm_model(model, path=None):
    if path is None:
        path = config.HMM_MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"HMM saved → {path}")


def load_hmm_model(path=None):
    if path is None:
        path = config.HMM_MODEL_PATH
    return joblib.load(path)


def save_encoder(encoder, path=None):
    if path is None:
        path = config.HMM_ENCODER_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(encoder, path)
    print(f"Encoder saved → {path}")


def load_encoder(path=None):
    if path is None:
        path = config.HMM_ENCODER_PATH
    return joblib.load(path)


def get_model_parameters(model):
    return {
        "start_prob":        model.startprob_,
        "transition_matrix": model.transmat_,
        "emission_matrix":   model.emissionprob_,
        "n_components":      model.n_components,
    }


def print_model_parameters(model, type_labels=None):
    params = get_model_parameters(model)
    state_labels = [config.HMM_STATE_LABELS.get(i, f"S{i}") for i in range(params["n_components"])]
    print("\n=== HMM Parameters ===")
    print("\nInitial state probabilities:")
    for i, p in enumerate(params["start_prob"]):
        print(f"  {state_labels[i]:20s}: {p:.4f}")
    print("\nTransition matrix (from→to):")
    header = "                     " + "  ".join(f"{s[:6]:>6}" for s in state_labels)
    print(header)
    for i, row in enumerate(params["transition_matrix"]):
        print(f"  {state_labels[i]:20s}" + "  ".join(f"{v:6.3f}" for v in row))
    if type_labels is not None:
        print("\nEmission probabilities (state × type):")
        for i, row in enumerate(params["emission_matrix"]):
            top = sorted(zip(row, type_labels), reverse=True)[:3]
            top_str = ", ".join(f"{t}:{p:.2f}" for p, t in top)
            print(f"  {state_labels[i]:20s}: {top_str}")


if __name__ == "__main__":
    from phase2_hmm.data_preprocessing import (
        load_glc_data, encode_features, build_observation_sequences
    )
    df = load_glc_data()
    df_enc, type_enc, trig_enc = encode_features(df)
    seqs, lengths, labels = build_observation_sequences(df_enc)
    n_symbols = len(type_enc.classes_)

    model = build_hmm(n_symbols=n_symbols)
    model = train_hmm(model, seqs, lengths)
    print_model_parameters(model, list(type_enc.classes_))
    save_hmm_model(model)
    save_encoder(type_enc)
    joblib.dump(trig_enc,
                os.path.join(config.CHECKPOINTS_DIR, "hmm_trigger_encoder.pkl"))
    print("All saved.")
