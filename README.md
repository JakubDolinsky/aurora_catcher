1. Project Overview
Aurora Catcher is a machine learning application designed to detect aurora (northern and southern lights) in photographs of the night sky.
The system helps users determine whether aurora is present in an image, especially in cases where it is faint or visually similar to other phenomena such as airglow, 
light pollution, or the Milky Way. The application is built as a complete end-to-end system with a graphical interface, allowing users to upload images and receive 
real-time predictions.

2. Key Features
Detection of aurora presence in night sky images
Classification of visually similar sky phenomena
Uncertainty-aware decision making (avoids false positives/negatives in ambiguous cases)
Fully interactive GUI for non-technical users
Modular ML inference pipeline

3. System Architecture
The system is composed of three main layers:

Presentation Layer (GUI):
Handles image upload and user interaction
Displays prediction results
Allows saving outputs

Mid Layer (Orchestration):
Coordinates model execution
Processes and formats predictions
Handles decision logic between models

ML Layer (Core Models)

The system uses a two-stage model pipeline:

Model 1: Detects whether aurora is present in the image
Model 2: Activated only if aurora is not detected; identifies other possible sky phenomena

This design improves reliability by separating high-precision detection from secondary classification tasks.

4. ML Pipeline
User uploads an image via GUI
Image is validated and preprocessed
Model 1 predicts aurora presence
If aurora is not detected, Model 2 is executed
Results are processed and displayed to the user

5. Dataset & Training
The models were trained on a dataset of over 16,000 night sky images containing arora and other multiple astronomical and atmospheric phenomena.

Key characteristics:

Binary classification for aurora detection
Multi-class classification for non-aurora phenomena (classes such as: airglow, milky way, zodiacal light, twilight etc.)

Augmented dataset to improve robustness
Train / validation / test split with additional hard-evaluation samples

6. Model Design
Both models are convolutional neural networks optimized for image classification.

Design highlights:

CNN-based architecture
Batch normalization and regularization techniques
Optimized using AdamW optimizer
Early stopping and learning rate scheduling

Model 1 is optimized for high-precision aurora detection, while Model 2 focuses on broader multi-class classification of similar sky phenomena.

7. Reliability & Decision Logic
To improve robustness in ambiguous cases, the system uses confidence-based decision logic:

Clear predictions when confidence is high
“Uncertain” state in borderline cases
Reduced risk of false aurora detection

This approach improves real-world usability, especially for faint or visually ambiguous aurora events.

8. Results (Summary)
Model performance:

Model 1 (Aurora detection)
High accuracy on standard test data, with expected performance drop on extreme edge cases
Model 2 (Phenomena classification)
Strong multi-class classification performance across 7 sky phenomena categories

9. How to Use
Run the application executable (located  in aurora_catcher\aurora catchern application\dist\AuroraCatcher)
Upload a supported image (night sky photo)
Click “Detect Aurora”
View results in the output window
Optionally save results as a text file

10. Limitations
The system is designed for night sky photography only.

Performance may degrade when:

- images contain daylight scenes or other scenes than night sky scenes
- input quality is very low or heavily distorted

11. Future Improvements
Deployment as web API
Mobile application support
Dataset expansion for rare aurora cases
Improved detection of extremely faint aurora events
Extension to broader astronomical scene classification

For better technical detail aurora catchern application\docs\technical.txt

You can try out demo application here: https://huggingface.co/spaces/kubizmus/aurora_catcher