Documentation for "Aurora catcher" application 

1. Brief Description

a) As solar activity is currently approaching its peak, many people are turning their eyes and cameras toward the sky in hopes of witnessing one of the most fascinating 
phenomena — the aurora, also known as the northern lights in the Northern Hemisphere. In many cases, the presence of aurora is not obvious and can easily be confused with 
similar phenomena such as airglow or light pollution. To help users better determine whether aurora is present in their photos, I created this neural network–based application.

b) Aurora can often be quite faint, making it difficult for people to clearly recognize and confirm its presence. The main goal of the application is to detect aurora in 
photographs of the night or evening sky. If no real aurora is detected, the application also informs users about other phenomena that may resemble aurora and appear in their 
images.

c) The core of the application is a machine learning layer consisting of two models. The first model determines whether aurora is present in the image. If no aurora is detected,
the second model is triggered to identify the three most visually similar phenomena. The mid layer orchestrates the models and collects their outputs, while the presentation 
layer translates the results and displays them in the GUI.

d) The application is intended primarily for inexperienced users, but it can also assist experienced observers in ambiguous situations. It is particularly useful when aurora is 
faint or when other phenomena, such as airglow, closely resemble its appearance.

2. Application architecture

a) The presentation layer is implemented as a desktop GUI. This layer is responsible for loading an image (the “Select image” button), running image analysis (the “Detect aurora”
 button), and presenting model outputs in a pop-up window. At the bottom of the pop-up window, there is a button for saving the result as a txt file.

b) The mid layer sits between the models and the presentation layer. It runs the models, translates their outputs, and prepares them for the presentation layer.

c) The core layer (NN layer) contains two convolutional neural networks that are chained into a pipeline by the mid layer. The first model detects aurora in the image. 
If it confirms aurora, the mid layer takes the output and stops further processing. If no aurora is detected, the mid layer triggers the second CNN model to detect other 
sky phenomena. The mid layer then processes the output and passes it to the presentation layer.

Pipeline

1. The image is loaded in the presentation layer.
2. Validation is performed in the presentation layer (maximum size, valid image format).
3. The image is preprocessed for inference. It is resized so that the longer side is 256 px, then transformations are applied (padding to make the image square and normalization).
4. The mid layer runs the first model, or both models depending on the result of the first model.
5. The mid layer collects and prepares the model output(s), and the translator converts them for the presentation layer.
6. The presentation layer displays the result in a pop-up window with the option to save the output.

3.Explanation for 2 model pipeline
The most important reason for using a two-model pipeline is the high specialization of the first model in recognizing aurora. The model’s verdict on whether aurora is present 
in the image should be as precise as possible, so it is important to keep the task simple. A single multilabel model that also includes aurora images would be more complex and 
would reduce the likelihood of achieving such high precision.

The main goal of the application is aurora detection. The second model therefore only determines which other phenomena could be present in the image. This is secondary information,
and its precision does not need to be as high as that of the first model.

4.Dataset
a) The dataset was created by collecting images from public sources, website and forum galleries, and personal galleries. Many of the images are protected by copyright, so it is
not possible to publish the full dataset. The images were used solely for training and validation purposes.
b)classes:
model1 : 2 classes (aurora, non aurora)
model2 : 7 classes (airglow, light pollution, lightning, milky way, NLC, twilight, zodiacal light)
c)dataset used for training both models contains 16333 images. 
d)preprocessing:
resize (128x128 for tunning hyperparameters, 256x256 for final training)
padding to square size,
normalization,
augmentation in training process: rotation (5 degrees, 20% data)
								  color jitter (brightness = 0.12, contrast = 0.12, 30% data)
								  gaussian noise (0.005, 20% data)
d) split: dataset has been split into three groups (train 70%, validation 15%, test 15%) and cca 90 images for hard validation check. 
This augmentation has been applied for both models.

5. Model architecture
Both models have similar architecture using CNN. 
CNN layer:
imput: 256x256
batch normalization
spatial dropout on some cnn layers and dropout on dense layer
RELU activation on cnn layers and sigmoid activation on dense layer

optimizer: AdamW
loss function:BCEWithLogitsLoss (binary cross entropy logits)
lr scheduler

best hyperparameters:
model1:
BATCH_SIZE = 16
EPOCHS = 80
LR = 0.0003 (learning rate)
WEIGHT_DECAY = 0.0001 (used in AdamW optimization)
SCHEDULER_PATIENCE = 6 (patience for scheduler lr improvement)
PATIENCE = 12 (patience for early stopping)

model2:
BATCH_SIZE = 16
EPOCHS =100
LR = 0.0003
WEIGHT_DECAY = 0.0001
SCHEDULER_PATIENCE = 6
PATIENCE = 12

training metrics for model1:
Train | loss=0.0931, accuracy=0.967, F1=0.965, Precision=0.980, Recall=0.950
Val   | loss=0.0685, accuracy=0.980, F1=0.978, Precission=0.992, Recall=0.965

training metrics for model2:
Train | loss=0.1437, accuracy=0.941, F1=0.847, Precision=0.860, Recall=0.869, M_PROBS (NN output probabilities per class) = airglow:0.129, 
light pollution:0.397, lightning:0.129, milky way:0.266, NLC:0.127, twilight:0.273, zodiacal light:0.088 
Val   | loss=0.1383, accuracy=0.944, F1=0.861, Precision=0.874, Rrecall=0.883, M_PROBS (NN output probabilities per class) = airglow:0.118, 
light pollution:0.405, lightning:0.137, milky way:0.266, NLC:0.118, twilight:0.261, zodiacal light:0.094

6. Thresholds
The output from the model is either a single probability value (from a sigmoid function) or a vector of probabilities per class. I wanted the application to provide users with
relevant and honest information about aurora occurrence. It should not falsely confirm aurora when it is not present, and it should also not reject aurora if it is actually there.

For Model 1, I defined probability intervals to improve the informational value of the result. For values below the lower threshold, the application returns false. 
For values above the upper threshold, it returns true. For values between these thresholds, the application returns uncertain. This covers cases where it is difficult or 
impossible to clearly confirm whether aurora is present.

For Model 2, there is no interval like in Model 1. Instead, there is a single threshold: probabilities above it indicate the presence of a given phenomenon. 
To provide more informative feedback, the application also includes the confidence level for each output.

Decision confidence thresholds fo model1:
(0.15, 0.75)

Decision thresholds for model2:
(
    0.3600,  # airglow
    0.3200,  # light pollution
    0.3800,  # lightning
    0.5000,  # milky way
    0.4600,  # NLC
    0.2600,  # twilight
    0.4100,  # zodiacal light
)

thresholds for confidence levels:
(
    [0.3600, 0.9342, 0.9795],  # airglow
    [0.3200, 0.8194, 0.9368],  # light pollution
    [0.3800, 0.9864, 0.9971],  # lightning
    [0.5000, 0.9193, 0.9692],  # milky way
    [0.4600, 0.9961, 0.9984],  # NLC
    [0.2600, 0.9767, 0.9942],  # twilight
    [0.4100, 0.7018, 0.8013],  # zodiacal light
)

7. application structure: 
aurora_catcher/
├── application/
│   ├── cnn_layer/
│   │   ├── model1_inference_engine/
│   │   │   ├── model_weights/
│   │   │   │   └── best_model.pt
│   │   │   └── model1_inference_engine.py
│   │   ├── model2_inference_engine/
│   │   │   ├── model_weights/
│   │   │   │   └── best_model.pt
│   │   │   └── model2_inference_engine.py
│   │   └── models/
│   │       ├── model1/
│   │       │   └── cnn_model.py
│   │       └── model2/
│   │           └── cnn_model.py
│   ├── common/
│   │   ├── data_preprocessing.py
│   │   ├── image_preprocess.py
│   │   └── logger.py
│   ├── log/  (saved log files)
│   ├── mid_layer/
│   │   └── mid_layer.py
│   ├── presentation_layer/
│   │   ├── presentation_layer_console.py
│   │   ├── presentation_layer_gui.py
│   │   └── translator.py
│   └── config.py
├── model1/
│   ├── common/
│   │   └── data_preprocessing.py
│   ├── dataset/  (dataset structure - aurora and non aurora)
│   ├── model/
│   │   ├── checkpoints/  (save the best training results)
│   │   ├── set_decision_threshold/
│   │   │   ├── best_model_pt
│   │   │	│	└──best_model.pt
│   │   │   ├── result/  (result from scripts - comparing predictions vs. real values)
│   │   │   ├── adjust_decision_threshold_interval.py
│   │   │   └── set_decision_threshold.py
│   │   ├── aurora_dataset.py
│   │   ├── cnn_model.py
│   │   └── cnn_train.py
│   ├── tests/
│   │   ├── manual_tests/
│   │   │   ├── dataset_sample_visualisation_test.py
│   │   │   ├── overfit_debug_test.py
│   │   │   └── visualisation_augmentation_test.py
│   │   ├── trained_model_tests/
│   │   │   ├── best_model_pt
│   │   │	│	└──best_model.pt
│   │   │   ├── test_results/  (test metrices)
│   │   │   └── trained_model_test.py
│   │   └── unit_tests/
│   │       ├── dataset_test.py
│   │       ├── model_test.py
│   │       ├── run_tests.py
│   │       └── train_test.py
│   └── config.py
├── model2/
│   ├── common/
│   │   └── data_preprocessing.py
│   ├── dataset/
│   │   └── pseudolabelling/
│   │       ├── model/
│   │       │   ├── checkpoints/  (best training results for multilabelling)
│   │       │   ├── model_for_labelling
│   │       │   │   └── best_model.pt
│   │       │   ├── cnn_train.py
│   │       │   ├── pseudolabelling.py
│   │       │   └── (csv with pseudolabelling results)
│   │       └── tests/
│   │           ├── manual_tests/
│   │           │   ├── dataset_sample_visualisation_test.py
│   │           │   ├── overfit_debug_test.py
│   │           │   └── visualisation_augmentation_test.py
│   │           ├── trained_model_tests/
│   │           │   ├── best_model_pt  
│   │           │   │	└── best_model.pt
│   │           │   └── trained_model_test.py
│   │           └── unit_tests/
│   │               ├── dataset_test.py
│   │               ├── model_test.py
│   │               ├── run_tests.py
│   │               └── train_test.py
│   ├── model/
│   │   ├── checkpoints/  (best results from training)
│   │   ├── set_decision_threshold/
│   │   │   ├── result/  (result from finding confidence levels of model2)
│   │   │   ├── best_model_pt  
│   │   │   │	└── best_model.pt
│   │   │   └── adjust_positive_threshold_and_prob_levels.py
│   │   ├── aurora_dataset.py
│   │   ├── cnn_model.py
│   │   └── cnn_train.py
│   ├── tests/
│   │   ├── manual_tests/
│   │   │   ├── dataset_sample_visualisation_test.py
│   │   │   ├── overfit_debug_test.py
│   │   │   └── visualisation_augmentation_test.py
│   │   ├── trained_model_test/
│   │   │   ├── best_model_pt  
│   │   │   │	└── best_model.pt
│   │   │   ├── result/  (test results - metrices)
│   │   │   └── trained_model_test.py
│   │   └── unit_tests/
│   │       ├── dataset_test.py
│   │       ├── model_test.py
│   │       ├── run_tests.py
│   │       └── train_test.py
│   └── config.py
├── .gitignore
└── README.md

				
8. The application accepts images with the following extensions: .jpg, .jpeg, .png, .bmp, and a maximum size of 6000 px on the longer side or 30 megapixels.
The model’s ability to recognize "zodiacal light" is slightly less reliable than for other classes due to the characteristics of the training dataset.

9. Future improvements:
Mobile application
API development
Front-end improvements
Enhancing the dataset for Model 2 to achieve more reliable results
Extending the application to recognize additional astronomical phenomena, enabling general recognition of phenomena in the evening or night sky
Using for classification of photos in personal dataset by sky phenomena