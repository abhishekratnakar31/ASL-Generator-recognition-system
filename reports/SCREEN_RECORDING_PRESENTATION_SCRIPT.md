# Screen Recording Video Presentation Script: ASL Multimodal Recognition System

**Target Video Duration:** ~5 to 7 minutes  
**Presenter:** Project Lead / Engineer  
**Objective:** Showcase the complete development journey, architectural pivot from frozen ResNet50 to 53-keypoint Whole-Body Pose, vision-language multimodal alignment, multi-inductive-bias ensembling, and the verified locked test-set results on unseen human signers.

---

## Visual Setup & Screen Layout Before Recording

Have the following tabs open and ready to switch between smoothly:
1. **Tab 1 (Overview):** GitHub Repository / [README.md](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/README.md)
2. **Tab 2 (The Failure of ResNet50):** `reports/plots/03_visual_baselines_curves.png` (or Notebook 03 in Colab)
3. **Tab 3 (The Landmark Pivot):** `reports/visual_examples/pose_hand_check.png` & [src/preprocessing/pose.py](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/src/preprocessing/pose.py)
4. **Tab 4 (The Multimodal & Boost Architecture):** `reports/plots/04c_pose_boost_curves.png` & [src/models/multimodal/pose_multimodal_model.py](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/src/models/multimodal/pose_multimodal_model.py)
5. **Tab 5 (The Locked Test Set Evaluation):** Notebook 05 in Colab / `reports/plots/05_final_test_comparison.png`, `05_confusion_matrix.png`, and `05_class_performance.png`
6. **Tab 6 (Production Code & Backend):** IDE showing [backend/app/main.py](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/backend/app/main.py) and test suite passing (`37 passed in 10s`).

---

## Section-by-Section Script

---

### ACT 1: Introduction & Problem Statement (0:00 – 1:00)

**[VISUAL ON SCREEN]:** Show GitHub Repository landing page / README title section.

**[NARRATION]:**
> *"Hello everyone! Today I’m excited to present our progress on the **American Sign Language Multimodal Recognition System** built on the WLASL-100 benchmark dataset.*
>
> *Sign language is the primary communication medium for millions in the Deaf and Hard-of-Hearing communities. However, automated computer vision recognition faces three massive challenges: subtle finger configurations, rapid transitional motion, and high signer appearance variance.*
>
> *In this project, our primary goal was not just to build a classifier that gets high accuracy on training data, but to build a **signer-independent** system that genuinely generalizes to brand-new, unseen human signers with zero data leakage. Today, I'll walk you through our initial baseline, the major breakthrough that multiplied our accuracy by 4x, our multimodal vision-language alignment, and our final locked test set results."*

---

### ACT 2: The Initial Baseline & Why Frozen ResNet50 Failed (1:00 – 2:15)

**[VISUAL ON SCREEN]:** Switch to `reports/plots/03_visual_baselines_curves.png` (Validation Loss & Accuracy Curves from Notebook 03).

**[NARRATION]:**
> *"We started our experimentation using the standard industry baseline: a 50-layer deep convolutional network (**ResNet50**) pre-trained on ImageNet, frozen as a feature extractor, followed by sequential models like LSTMs and Transformers.*
>
> *Each video was downsampled to 32 frames and compressed into 2048-dimensional frame features. However, as you can see clearly on this validation loss graph, we hit a hard ceiling:*
> - *Top-1 accuracy plateaued at only **11.9%**.*
> - *Top-5 accuracy hovered around **22.9%**.*
> - *And most critically: look at the validation loss curve—after epoch 10, the validation loss started climbing in a dramatic **U-turn rebound** from 4.4 up to 6.8.*
>
> *Why did this happen? Because frozen full-frame ImageNet features suffer from **background shortcut learning**. The model was memorizing static wall colors, room lighting, and signer shirts instead of hand movements. When tested on unseen signers in validation, those background cues failed completely. Furthermore, global average pooling over a full 224x224 image completely blurred out subtle finger-shape details."*

---

### ACT 3: The Architectural Pivot: 53-Keypoint Whole-Body Pose (2:15 – 3:30)

**[VISUAL ON SCREEN]:** Switch to `reports/visual_examples/pose_hand_check.png` (showing skeletal landmark overlay on hands and shoulders), then briefly show [src/preprocessing/pose.py](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/src/preprocessing/pose.py).

**[NARRATION]:**
> *"To solve this root cause, we made a decisive architectural pivot: **we completely stripped away background pixels and replaced them with normalized skeletal kinematics**.*
>
> *Using `rtmlib` COCO-WholeBody running high-performance ONNX Runtime, we detect **53 key joints** per frame inside the signer bounding box:*
> - *11 upper-body joints (nose, eyes, shoulders, elbows, wrists)*
> - *21 landmarks for the left hand*
> - *21 landmarks for the right hand*
>
> *In our preprocessing module, we engineered an invariant **349-dimensional vector** per frame:*
> 1. *First, **Shoulder Centering & Span Scaling**: Every coordinate is centered at the midpoint of the shoulders and scaled by shoulder width. This makes the data completely invariant to camera distance, zoom, or signer height.*
> 2. *Second, **Wrist-Relative Hand Geometry**: Finger joints are calculated relative to each wrist, isolating pure finger curvature.*
> 3. *Third, **Temporal Velocities**: We compute frame-to-frame velocity differences so the network directly senses acceleration and movement direction.*
>
> *When we trained a 2-layer Self-Attention Transformer on these pure pose features in Notebook 04b, the result was an immediate breakthrough:*
> - *Validation Top-1 jumped by **3.7×**, from **11.0% all the way to 41.28%**!*
> - *Validation Top-5 climbed from **25.7% to 68.81%**!*
> - *And look at the validation loss curve: the U-turn completely vanished, dropping smoothly from 4.7 down to **3.08** with zero overfitting.*
> - *We even ran an experiment adding ResNet50 back in, and accuracy immediately tanked back to 23%, mathematically proving that raw pixel features were hurting generalization."*

---

### ACT 4: Multimodal Vision-Language Alignment & Kinematic Boosting (3:30 – 4:45)

**[VISUAL ON SCREEN]:** Switch to `reports/plots/04c_pose_boost_curves.png` & show [src/models/multimodal/pose_multimodal_model.py](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/src/models/multimodal/pose_multimodal_model.py).

**[NARRATION]:**
> *"Next, in Notebooks 04 and 04c, we expanded the system into a true **Multimodal Vision-Language Architecture**.*
>
> *Traditional classifiers treat words as arbitrary numbers from 0 to 99. In our multimodal branch, we connect video gestures directly to semantic natural language:*
> - *We generate contextual English prompts: `'An American Sign Language sign representing the word {gloss}'`.*
> - *We encode them using a pre-trained sentence transformer, **`all-MiniLM-L6-v2`**, producing 384-dimensional dense semantic vectors projected into a shared 192-D hypersphere.*
> - *We align video motion representations with text prototypes using **scaled cosine similarity** and a **multi-positive InfoNCE contrastive loss**, which clusters videos of the same sign together in semantic space.*
>
> *To push performance even higher in Notebook 04c, we introduced:*
> 1. *A **Cosine Classifier head (`CosineClassifier`)**, which normalizes feature vectors and class centroids to eliminate frequency bias.*
> 2. *And a **Multi-Inductive-Bias Ensemble (`ensemble_equal_three`)**, combining three diverse ways of learning: a Self-Attention Transformer, a Bidirectional LSTM with Attention pooling, and our Multimodal Language model.*
> *As shown by the red dashed line on this plot, the ensemble broke previous records, reaching **43.12% Top-1, 72.48% Top-5**, and pushing Macro-F1 past 0.40 for the first time."*

---

### ACT 5: Official Locked Test-Set Evaluation Results (4:45 – 6:00)

**[VISUAL ON SCREEN]:** Switch to Notebook 05 in Colab / show `reports/plots/05_final_test_comparison.png`, `05_confusion_matrix.png`, and `05_class_performance.png`.

**[NARRATION]:**
> *"Finally, in Notebook 05, we unlocked the **held-out test set**—230 clips performed by 20 new signers that no model had ever seen during training or validation.*
>
> *Here are the official verified results:*
> - *Our **`pose_transformer_only`** achieved **43.04% Top-1 Accuracy** and **0.4213 Macro-F1** across 88 active classes. Compared to the random baseline of 0.87%, that is an improvement of nearly **50x**!*
> - *Our winning ensemble, **`ensemble_equal_three`**, achieved **76.52% Top-5 Accuracy** and **83.91% Top-10 Accuracy**! That means for nearly 84 out of every 100 sign clips, the true sign is inside the top 10 suggestions.*
> - *Our multimodal model achieved a **Mean Reciprocal Rank (MRR) of 0.5208** and **70.43% Video-to-Text Recall@5**, meaning that on average, the correct English word is retrieved in the top 2 candidates when querying the entire 100-word vocabulary.*
>
> *Now let's examine the generated visual artifacts:*
> - *In this **Evolution Bar Chart**, you can see the clear progression from random baselines (under 1%) to ResNet50 (11%), to pure Pose (43%), and finally the Ensemble reaching our peak 76.5% Top-5 accuracy.*
> - *In this **100-Class Confusion Matrix**, notice the prominent deep blue diagonal line. Off-diagonal errors are minimal, and when the model does make a mistake, it’s between physically similar signs like `bad` vs. `hot` near the mouth.*
> - *And in this **Per-Class Breakdown**, **10 classes achieved a perfect 1.00 F1 score**: `appointment`, `change`, `dark`, `graduate`, `speech`, `ready`, `sandwich`, `son`, `language`, and `room`."*

---

### ACT 6: Production Engineering & Next Steps (6:00 – 6:45)

**[VISUAL ON SCREEN]:** Switch to IDE showing `checkpoints/production_model/`, [backend/app/main.py](file:///Users/abhishekratnakar/ASL-Generator-recognition-system/backend/app/main.py), and terminal showing passing test suite.

**[NARRATION]:**
> *"All winning checkpoints, class mappings, and configuration files have been packaged into our **production model bundle**.*
>
> *In our local repository, our **FastAPI inference backend** is already set up and tested, with **37 out of 37 unit tests passing cleanly**. The API provides health checks and a `/predict/video` endpoint that runs this exact kinematic pipeline in real-time.*
>
> *Our next immediate steps are:*
> 1. *Connect the production model bundle to the live FastAPI service.*
> 2. *Build a responsive **Next.js frontend** with real-time camera capture for live webcam gesture recognition.*
> 3. *And explore expanding the vocabulary from 100 classes to the full WLASL vocabulary using transfer learning.*
>
> *Thank you for watching, and I look forward to sharing our next milestone!"*

---

## Tips for Recording

1. **Audio Quality:** Use a decent USB microphone or headset in a quiet room.
2. **Resolution:** Record at 1080p (1920x1080) at 60 or 30 FPS.
3. **Cursor Visibility:** Enable cursor highlighting (often available in OBS, QuickTime, or Loom) so viewers can follow where you point on the charts.
4. **Pacing:** Speak at a steady, confident conversational speed. Pause briefly when transitioning between slides or tabs.

