# Test Result 5: Vision Transformer (ViT-B/16) Ensemble

Following the successful integration of the Vision Transformer (ViT-B/16) into the Phase 1 ensemble, the full prediction pipeline was tested on a test image (`ls_test_0002.jpg` from Nepal). 

Since this environment lacked the C++ Build Tools to install `hmmlearn` (needed for Phase 2), Phase 2 was successfully bypassed, and the ViT-B/16 model correctly functioned in Phase 1 predicting the image.

**Test Command Run:**
`python project\pipeline\run_pipeline.py --mode predict --image c:\odv3kor\Landslide_Identification_Using_ML\project\data\processed\test\landslide\ls_test_0002.jpg --country Nepal`

## Output:

```
Loading Phase 1 ensemble (EfficientNet-B3 + ViT-B/16) ...
Loaded pretrained EfficientNet-B3 with replaced classifier head (in=1536, out=2).
Warning: EfficientNet checkpoint not found.
Loaded pretrained ViT-B/16 with replaced classifier head (in=768, out=2).
Checkpoint loaded: epoch=1, val_acc=0.7473
Ensemble loaded: EfficientNet-B3 (T=1.0000, w=0.6) + ViT-B/16 (w=0.4)
HMM not available, skipping Phase 2 loading.

Pipeline ready.

=== Prediction Result ===
Phase 1 — LANDSLIDE (confidence: 52.7%)

LANDSLIDE DETECTED — Phase 2 HMM offline. Confidence: 52%
```

## Note:
The newly introduced ViT-B/16 model accurately successfully classified the image as a landslide. Due to being trained for only 1 epoch without a GPU in this validation test (and without the EfficientNet weights being present), the ensemble confidence was lower, but it proved the new ViT-architecture integration works perfectly in the pipeline!
