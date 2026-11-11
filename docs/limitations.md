# Limitations and responsible use

- The model predicts dataset labels from images; it does not assess structural integrity or building safety.
- Image conditions, camera distance, lighting, texture, background and dataset bias can influence predictions.
- Similar-looking categories may be confused; report per-class metrics rather than accuracy alone.
- Dataset split leakage from duplicate or near-duplicate images has not yet been ruled out.
- Dataset provenance and licence must be checked before redistribution.
- Grad-CAM is qualitative and can highlight contextual regions; it is not a segmentation mask or proof that the model reasons like an inspector.
- A qualified building professional must make inspection and remediation decisions.
