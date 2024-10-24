# Leveraging-Deep-Learning-in-the-Detection-of-Malaria-in-Blood-Smear-Images
This project leverages Convolutional Neural Networks (CNNs) to automate malaria detection, enhancing diagnostic speed and accuracy. EfficientNetB0 outperformed other models, showing high accuracy and efficiency, making it suitable for resource-limited settings and potential mobile applications.
# Table of Content
1. Objectives
2. Methodology
3. Results
4. Performance Comparism
5. Conclusion
6. Recommendations

# Objectives
The primary objectives of the project were:
* To build a CNN-based malaria detection system that classifies blood smear images as parasitized or uninfected.
* To compare the performance of different CNN architectures and identify the most efficient model in terms of accuracy and computational efficiency.
* To optimize the models to ensure their applicability in resource-limited settings, where high accuracy and low computational requirements are crucial.

# Methodology
**Dataset**

The dataset comprised 27,560 blood smear images, evenly distributed between two classes:
* Parasitized: 13,780 images
* Uninfected: 13,780 images

**Data Preprocessing**

* All images were resized to 224x224 pixels to standardize input dimensions.
* Image augmentation techniques (rotation, zooming, horizontal flipping, etc.) were applied to improve generalization and reduce overfitting.
* Images were normalized by rescaling the pixel values to a range of [0,1].

**Model Development**

The project implemented and compared five CNN architectures:
* Manual CNN: A simple CNN model with three convolutional layers, batch normalization, max-pooling, and dense layers.
* ResNet50: A pre-trained ResNet50 model with custom top layers.
* VGG16: A VGG16 model with additional dense layers for binary classification.
* InceptionV3: A transfer learning model optimized for large-scale image classification.
* EfficientNetB0: An efficient and lightweight CNN optimized for high accuracy and low computational cost.

**Training Procedure**

* The models were trained using the Adam optimizer and binary cross-entropy loss function.
* Early stopping and learning rate reduction were employed to avoid overfitting.
* The training data was split into 80% for training, 10% for validation, and 10% for testing.

# Results

**Model Accuracy**

Each model's performance was evaluated based on accuracy, validation accuracy, loss, and additional metrics such as precision, recall, and F1-score. The results are summarized below:

**Model Test Accuracy(TA) & Validation Accuracy(VA)**
* Manual CNN - *( TA - 0.95*, *VA -  0.94 )*
* ResNet50 - *( TA - 0.94*, *VA - 0.95 )*
* VGG16 - *( TA - 0.94*, *VA - 0.94 )*
* InceptionV3 - *( TA - 0.92*, *VA - 0.95 )*
* EfficientNetB0 - *( TA - 0.95*,  *VA - 0.95 )*

**Model Evaluation**

The confusion matrices and ROC curves were generated for each model to further evaluate performance. These metrics showed high sensitivity and specificity in all models, with EfficientNetB0 and Manual CNN slightly outperforming the others in terms of accuracy.

**ROC Curve and AUC**

The Area Under the Curve (AUC) was calculated for each model to quantify their ability to distinguish between parasitized and uninfected cells. EfficientNetB0 showed the highest AUC score of 0.95, followed closely by Manual CNN and ResNet50.

# Performance Comparison

After analyzing the results, EfficientNetB0 emerged as the best model based on its balance of accuracy, computational efficiency, and validation performance. Although the Manual CNN also achieved comparable accuracy, EfficientNetB0's pre-trained nature and optimized architecture made it more suitable for deployment in real-world settings, especially in resource-limited environments.

# Conclusion

This project successfully demonstrated the power of CNN models in automating the detection of malaria in blood smear images. By comparing five different models, EfficientNetB0 was identified as the most promising model for practical use, due to its accuracy and efficient design. Future work could involve deploying this model in mobile applications to further improve the accessibility and efficiency of malaria diagnosis in remote areas.

# Recommendations

* Deployment of the selected model (EfficientNetB0) in mobile or embedded systems for real-time malaria detection.
* Expanding the dataset to include more diverse blood smear images from different geographical regions to improve generalization.
* Investigation into further optimization techniques such as model pruning and quantization to reduce the model's size and computational requirements for deployment in low-resource settings.
