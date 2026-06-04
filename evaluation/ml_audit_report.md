# Deepfake Detector Model Audit Report

## Model Metadata
* **Model File Path:** `D:\deepfake_detector\saved_models\best_model.keras`
* **Checkpoint File Used:** `best_model.keras`
* **Model Creation Date:** `2026-06-03 00:39:02`
* **Model Last Modified:** `2026-06-03 01:08:13`
* **Model File Size:** `52.25 MB`
* **Streamlit App Path Check:** `MATCH: Streamlit app loads 'saved_models/best_model.keras'`

## Verification of Class Mapping
* **Class 0 (Real):** Verified that files from the `Real` folders are labeled as `0`.
* **Class 1 (Fake):** Verified that files from the `Fake` folders are labeled as `1`.
* **Decision Boundary:** Threshold >= `0.5` predicts Fake (1), and < `0.5` predicts Real (0).

## Evaluation Metrics (Entire Test Dataset)
* **Dataset Size:** 10905 images (5413 Real, 5492 Fake)
* **Training Accuracy (Subset of 4000):** `86.52%`
* **Validation Accuracy (Subset of 1000):** `75.80%`
* **Test Accuracy (Entire Dataset of 10905):** `69.39%`
* **Precision:** `66.89%`
* **Recall:** `77.68%`
* **F1 Score:** `71.88%`
* **ROC-AUC:** `0.773434`

### Confusion Matrix
| | Predicted Real (0) | Predicted Fake (1) |
|---|---|---|
| **Actual Real (0)** | 3301 | 2112 |
| **Actual Fake (1)** | 1226 | 4266 |

---

## Correctly Classified Test Samples (20 Random Samples)
| File | Actual Class | Predicted Class | Raw Probability Output (Fake) |
|---|---|---|---|
| `fake_840.jpg` | Fake (1) | Fake (1) | `0.990250` |
| `real_287.jpg` | Real (0) | Real (0) | `0.193430` |
| `real_4351.jpg` | Real (0) | Real (0) | `0.336941` |
| `real_1836.jpg` | Real (0) | Real (0) | `0.292757` |
| `fake_4998.jpg` | Fake (1) | Fake (1) | `0.502476` |
| `fake_523.jpg` | Fake (1) | Fake (1) | `0.591862` |
| `real_1075.jpg` | Real (0) | Real (0) | `0.316930` |
| `real_4306.jpg` | Real (0) | Real (0) | `0.374644` |
| `real_1336.jpg` | Real (0) | Real (0) | `0.186226` |
| `real_4319.jpg` | Real (0) | Real (0) | `0.356927` |
| `fake_384.jpg` | Fake (1) | Fake (1) | `0.898984` |
| `real_2727.jpg` | Real (0) | Real (0) | `0.236125` |
| `real_3521.jpg` | Real (0) | Real (0) | `0.403317` |
| `fake_1032.jpg` | Fake (1) | Fake (1) | `0.685596` |
| `fake_4934.jpg` | Fake (1) | Fake (1) | `0.997270` |
| `fake_547.jpg` | Fake (1) | Fake (1) | `0.998856` |
| `real_3729.jpg` | Real (0) | Real (0) | `0.243522` |
| `real_2439.jpg` | Real (0) | Real (0) | `0.079348` |
| `fake_4417.jpg` | Fake (1) | Fake (1) | `0.753091` |
| `real_5263.jpg` | Real (0) | Real (0) | `0.367797` |

## Incorrectly Classified Test Samples (20 Random Samples)
| File | Actual Class | Predicted Class | Raw Probability Output (Fake) |
|---|---|---|---|
| `fake_1485.jpg` | Fake (1) | Real (0) | `0.494794` |
| `fake_4732.jpg` | Fake (1) | Real (0) | `0.326746` |
| `real_1888.jpg` | Real (0) | Fake (1) | `0.515897` |
| `real_4456.jpg` | Real (0) | Fake (1) | `0.504663` |
| `fake_3573.jpg` | Fake (1) | Real (0) | `0.302347` |
| `fake_322.jpg` | Fake (1) | Real (0) | `0.336449` |
| `fake_4333.jpg` | Fake (1) | Real (0) | `0.463892` |
| `real_3924.jpg` | Real (0) | Fake (1) | `0.521061` |
| `real_1523.jpg` | Real (0) | Fake (1) | `0.874545` |
| `real_2408.jpg` | Real (0) | Fake (1) | `0.565099` |
| `real_4522.jpg` | Real (0) | Fake (1) | `0.752181` |
| `fake_4923.jpg` | Fake (1) | Real (0) | `0.483755` |
| `real_3159.jpg` | Real (0) | Fake (1) | `0.614063` |
| `real_2310.jpg` | Real (0) | Fake (1) | `0.746792` |
| `fake_220.jpg` | Fake (1) | Real (0) | `0.355395` |
| `real_3691.jpg` | Real (0) | Fake (1) | `0.501042` |
| `fake_3944.jpg` | Fake (1) | Real (0) | `0.205439` |
| `fake_1147.jpg` | Fake (1) | Real (0) | `0.381659` |
| `fake_290.jpg` | Fake (1) | Real (0) | `0.490315` |
| `real_1113.jpg` | Real (0) | Fake (1) | `0.633809` |
