1. Project Overview
Aurora Catcher is a machine learning application designed to detect aurora (northern and southern lights) in photographs of the night sky.
The system helps users determine whether aurora is present in an image, especially in cases where it is faint or visually similar to other phenomena such as airglow, 
light pollution, or the Milky Way. The application is built as a complete end-to-end system with a graphical interface, allowing users to upload images and receive 
real-time predictions.

2. Problem and solution
Sometimes it can be tricky to distinguish between aurora and similar phenomena in the night sky, or whether aurora is present. To solve this problem I designed
two-stage computer vision pipeline.

3. ML Pipeline
a) user uploads an image via GUI
b) image is validated and preprocessed
c) model 1 predicts aurora presence
d) if aurora is not detected, Model 2 is executed and identifies other possible sky phenomena
e) results are processed and displayed to the user

4. Model Design
Both models are convolutional neural networks optimized for image classification.
Model 1 is specialized for high-precision aurora detection, while Model 2 focuses on broader multi-class classification of similar sky phenomena.

6. Key design decisions:
- two-staged inference pipeline (two separated models strictly specialized for each task: detecting aurora and distinguishing other phenomena than aurora)
- adding uncertainty interval for final decisions (model says "I do not know" rather than making false positive/negative decisions if the signal is low)

7. Dataset & Training
The models were trained on a dataset of over 16,000 night sky images containing aurora and other various astronomical and atmospheric phenomena. Dataset was
built from publicly available photography sources and astronomy communities. Dataset was then cleaned and processed for 
training purpose. 

The key part of the project was data labeling. Dataset for model 1 was created by collecting separate aurora and non-aurora images leading into binary labels.
The non-aurora dataset was created by collecting images according to dominant phenomenon. Therefore, this dataset was partially labeled. To complete the labels 
I used pseudolabeling approach. I trained the first version of model 2 using incompletely labeled data and then I used it to predict missing labels. Finally
I reviewed results, completed and corrected missing or incorrect labels.

Key characteristics:

Binary classification for aurora detection
Multi-class classification for non-aurora phenomena (classes such as: airglow, milky way, zodiacal light, twilight etc.)

Augmented dataset to improve robustness
Train / validation / test split with additional hard-evaluation samples

8.Results
Test and hard validation metrics for model 1 and model 2.

Model 1:
Accuracy: 0.9796
Precision: 0.9850
Recall: 0.9714 
F1 score: 0.9781

HARD VALIDATION dataset:
Accuracy: 0.8358
F1 score: 0.7317

Model 2:
Accuracy: 0.947
Precision: 0.852
Recall: 0.898
F1 score: 0.873

HARD VALIDATION dataset: 
Accuracy: 0.886
F1 score: 0.725

9. Limitations
The system is designed for night sky photography only.

Performance may degrade when:

- images contain daylight scenes or other scenes than night sky scenes
- input quality is very low or heavily distorted

For more technical details: docs\technical.txt

You can try out demo application here: https://huggingface.co/spaces/kubizmus/aurora_catcher (upload pictures of night sky containing or not containing aurora)

Github repo: https://github.com/JakubDolinsky/aurora_catcher.git