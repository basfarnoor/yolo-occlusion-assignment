# YOLO Occlusion-Sensitivity Run Summary

- Model: `yolo26n.pt` (pretrained, prediction only -- no training/fine-tuning)
- Ultralytics package version: `8.4.105`
- Python version: `3.14.6`
- Device: `cpu`
- Image size: `640`
- Confidence threshold: `0.05`
- Images processed: **30**
- Images failed: **0**
- Total runtime: **5.43 s**
- Average runtime per image: **181.1 ms**
- Annotated results: `results\annotated/<sample>/`

This is a raw-detection count, not a measure of correctness. A box appearing in the output means the model produced a prediction above the confidence threshold -- it does not by itself mean the prediction is right. Task 5 manually reviews one target object per sample to judge correctness.
