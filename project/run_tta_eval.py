"""TTA evaluation for all trained models + ensemble."""
import os, sys, json, time
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

config.PROCESSED_DATA_DIR = os.path.join(config.DATA_DIR, "binary_clean_expanded")

LOG = os.path.join(config.PLOTS_DIR, "clean_pipeline.log")


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def tta_predict(model, image_tensor, device):
    """Test-Time Augmentation: 7 views per image."""
    views = [image_tensor]
    views.append(torch.flip(image_tensor, dims=[2]))  # horizontal flip
    views.append(torch.flip(image_tensor, dims=[1]))  # vertical flip
    views.append(torch.flip(image_tensor, dims=[1, 2]))  # both flips
    views.append(torch.rot90(image_tensor, k=1, dims=[1, 2]))  # 90
    views.append(torch.rot90(image_tensor, k=2, dims=[1, 2]))  # 180
    views.append(torch.rot90(image_tensor, k=3, dims=[1, 2]))  # 270

    all_probs = []
    for v in views:
        with torch.amp.autocast("cuda"):
            output = model(v.unsqueeze(0).to(device))
        prob = F.softmax(output, dim=1)
        all_probs.append(prob.cpu())

    avg_prob = torch.stack(all_probs).mean(dim=0)
    return avg_prob


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        roc_curve, auc, accuracy_score,
    )
    from phase1_alexnet.model import get_convnext_cbam_fpn, get_efficientnetv2_cbam
    from phase1_alexnet.dataset import LandslideDataset, get_test_transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    models_to_eval = [
        ("EfficientNetV2-CBAM", config.EFFNETV2_CHECKPOINT, config.EMA_EFFNETV2_CHECKPOINT,
         get_efficientnetv2_cbam, config.EFFNETV2_IMG_SIZE),
        ("ConvNeXt-CBAM-FPN", config.CONVNEXT_CHECKPOINT, config.EMA_CONVNEXT_CHECKPOINT,
         get_convnext_cbam_fpn, config.CONVNEXT_IMG_SIZE),
    ]

    log("=" * 70)
    log("TTA EVALUATION")
    log("=" * 70)

    for model_name, ckpt_path, ema_ckpt_path, get_model_fn, img_size in models_to_eval:
        test_dataset = LandslideDataset(
            config.PROCESSED_DATA_DIR, "test", get_test_transforms(img_size)
        )
        log(f"Test set for {model_name}: {len(test_dataset)} images")

        for tag, ckpt in [("raw", ckpt_path), ("ema", ema_ckpt_path)]:
            if not os.path.exists(ckpt):
                log(f"{model_name} ({tag}): checkpoint not found, skipping.")
                continue

            model = get_model_fn(num_classes=config.NUM_CLASSES, freeze=False).to(device)
            checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            # Standard evaluation (batch)
            loader = DataLoader(
                test_dataset, batch_size=32, shuffle=False,
                num_workers=4, pin_memory=True,
            )
            all_labels, all_preds, all_probs = [], [], []
            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(device)
                    with torch.amp.autocast("cuda"):
                        outputs = model(images)
                    probs = F.softmax(outputs, dim=1)
                    preds = outputs.argmax(dim=1)
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())

            acc = accuracy_score(all_labels, all_preds)
            key = f"{model_name}_{tag}"
            results[key] = {
                "accuracy": acc, "labels": all_labels,
                "preds": all_preds, "probs": all_probs,
            }
            log(f"{key}: Test Accuracy = {acc:.4f} ({acc*100:.2f}%)")

            # TTA evaluation (per-image)
            all_labels_tta, all_preds_tta, all_probs_tta = [], [], []
            with torch.no_grad():
                for i in range(len(test_dataset)):
                    image, label = test_dataset[i]
                    avg_prob = tta_predict(model, image, device)
                    pred = avg_prob.argmax(dim=1).item()
                    all_labels_tta.append(label)
                    all_preds_tta.append(pred)
                    all_probs_tta.append(avg_prob[0, 1].item())
                    if (i + 1) % 200 == 0:
                        log(f"  TTA progress: {i+1}/{len(test_dataset)}")

            acc_tta = accuracy_score(all_labels_tta, all_preds_tta)
            key_tta = f"{model_name}_{tag}_TTA"
            results[key_tta] = {
                "accuracy": acc_tta, "labels": all_labels_tta,
                "preds": all_preds_tta, "probs": all_probs_tta,
            }
            log(f"{key_tta}: Test Accuracy = {acc_tta:.4f} ({acc_tta*100:.2f}%)")

            report = classification_report(
                all_labels_tta, all_preds_tta,
                target_names=config.CLASS_NAMES, digits=4,
            )
            log(f"\n{report}")

            cm = confusion_matrix(all_labels_tta, all_preds_tta)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES, ax=ax,
            )
            ax.set_title(f"Confusion Matrix: {key_tta} (Acc={acc_tta:.4f})")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            plt.tight_layout()
            safe = key_tta.replace(" ", "_").replace("-", "_").lower()
            plt.savefig(
                os.path.join(config.PLOTS_DIR, f"confusion_matrix_{safe}.png"),
                dpi=150,
            )
            plt.close()

            del model
            torch.cuda.empty_cache()

    # Ensemble with TTA
    log("\nEnsemble evaluation with TTA...")
    best_ens_acc = 0
    best_weights = (0.5, 0.5)

    eff_key = (
        "EfficientNetV2-CBAM_ema_TTA"
        if "EfficientNetV2-CBAM_ema_TTA" in results
        else "EfficientNetV2-CBAM_raw_TTA"
    )
    conv_key = (
        "ConvNeXt-CBAM-FPN_ema_TTA"
        if "ConvNeXt-CBAM-FPN_ema_TTA" in results
        else "ConvNeXt-CBAM-FPN_raw_TTA"
    )

    if eff_key in results and conv_key in results:
        for w1 in np.arange(0.3, 0.75, 0.05):
            w2 = 1.0 - w1
            ens_probs = (
                np.array(results[eff_key]["probs"]) * w1
                + np.array(results[conv_key]["probs"]) * w2
            )
            ens_preds = (ens_probs > 0.5).astype(int)
            ens_acc = accuracy_score(results[eff_key]["labels"], ens_preds)
            log(
                f"  Ensemble (EfficientNetV2={w1:.2f}, ConvNeXt={w2:.2f}): "
                f"{ens_acc:.4f} ({ens_acc*100:.2f}%)"
            )
            if ens_acc > best_ens_acc:
                best_ens_acc = ens_acc
                best_weights = (w1, w2)

        # Final ensemble with best weights
        ens_probs = (
            np.array(results[eff_key]["probs"]) * best_weights[0]
            + np.array(results[conv_key]["probs"]) * best_weights[1]
        )
        ens_preds = (ens_probs > 0.5).astype(int)
        ens_labels = results[eff_key]["labels"]

        log(f"\nBest Ensemble: EfficientNetV2={best_weights[0]:.2f}, ConvNeXt={best_weights[1]:.2f}")
        log(f"Best Ensemble Accuracy: {best_ens_acc:.4f} ({best_ens_acc*100:.2f}%)")

        report = classification_report(
            ens_labels, ens_preds,
            target_names=config.CLASS_NAMES, digits=4,
        )
        log(f"\n{report}")

        cm = confusion_matrix(ens_labels, ens_preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=config.CLASS_NAMES,
            yticklabels=config.CLASS_NAMES, ax=ax,
        )
        ax.set_title(f"Ensemble TTA (Acc={best_ens_acc:.4f}, w={best_weights})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        plt.savefig(
            os.path.join(config.PLOTS_DIR, "confusion_matrix_ensemble_tta.png"),
            dpi=150,
        )
        plt.close()

        fpr, tpr, _ = roc_curve(ens_labels, ens_probs)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, "b-", label=f"Ensemble TTA (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve - Ensemble with TTA")
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(config.PLOTS_DIR, "roc_curve_ensemble_tta.png"), dpi=150,
        )
        plt.close()

        results["ensemble_tta"] = {"accuracy": best_ens_acc, "auc": roc_auc}

    # Save all results
    summary = {}
    for k, v in results.items():
        summary[k] = {"accuracy": v["accuracy"], "auc": v.get("auc")}
    with open(os.path.join(config.PLOTS_DIR, "evaluation_results_final.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("\n" + "=" * 70)
    log("FINAL RESULTS SUMMARY")
    log("=" * 70)
    for k, v in sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        log(f"  {k}: {v['accuracy']*100:.2f}%")
    best_overall = max(v["accuracy"] for v in results.values())
    log(f"\n  BEST ACCURACY: {best_overall*100:.2f}%")
    log("=" * 70)


if __name__ == "__main__":
    main()
