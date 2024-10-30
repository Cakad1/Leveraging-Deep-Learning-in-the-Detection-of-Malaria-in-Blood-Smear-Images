# Leveraging-Deep-Learning-in-the-Detection-of-Malaria-in-Blood-Smear-Images

This project leverages Convolutional Neural Networks (CNNs) to automate malaria detection, enhancing diagnostic speed and accuracy. EfficientNetB0 outperformed other models, showing high accuracy and efficiency, making it suitable for resource-limited settings and potential mobile applications.	

## Table of Contents
1. [Objectives](#objectives)
2. [Methodology](#methodology)
3. [Results](#results)
4. [Performance Comparison](#performance-comparison)
5. [Conclusion](#conclusion)
6. [Recommendations](#recommendations)

## Objectives

The primary objectives of the project were:
* To build a CNN-based malaria detection system that classifies blood smear images as parasitized or uninfected.
* To compare the performance of different CNN architectures and identify the most efficient model in terms of accuracy and computational efficiency.
* To optimize the models to ensure their applicability in resource-limited settings, where high accuracy and low computational requirements are crucial.

## Methodology

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

## Results

**Model Accuracy**

Each model's performance was evaluated based on accuracy, validation accuracy, loss, and additional metrics such as precision, recall, and F1-score. The results are summarized below:

**Model Test Accuracy & Validation Accuracy**

![Compare CNN Model Performance](https://github.com/user-attachments/assets/ce084596-1528-493b-8de7-2e7f35679299)

**Model Evaluation**

The confusion matrices and ROC curves were generated for each model to further evaluate performance. These metrics showed high sensitivity and specificity in all models, with EfficientNetB0 and Manual CNN slightly outperforming the others in terms of accuracy.

*Manual CNN*

![Manual CNN Classification Report](https://github.com/user-attachments/assets/38201cc3-90e5-4383-8590-1f0d4b07e8bd)

![Manual CNN Confusion Matrix](https://github.com/user-attachments/assets/a369fffb-6c8d-438c-ba8e-f9378265a075)

![Manual CNN Roc Auc](https://github.com/user-attachments/assets/3da1cc7b-14df-44a3-b2ce-6ef61f55a392)

*ResNet50*

![ResNet50 Classification Report](https://github.com/user-attachments/assets/32935bde-c1c1-440d-bf3d-ea55ef83ff69)

![ResNet50 Confusion Matrix](https://github.com/user-attachments/assets/8acdf52c-8965-4ffd-be35-65365ddbe0a8)

![ResNet50  Roc Auc](https://github.com/user-attachments/assets/a7606170-83e9-43c0-a236-739ced332181)

*VGG16*

![VGG16 Classification Report](https://github.com/user-attachments/assets/ab03729e-a097-4a9c-9bce-833c2adf2f8e)

![VGG16 Confusion Matrix](https://github.com/user-attachments/assets/1f8c701a-4a94-4208-b4d5-3141d39815ff)

![VGG16 Roc Auc](https://github.com/user-attachments/assets/858f7752-71c7-4821-82cc-92b2c17c25ad)

*InceptionV3*

![InceptionV3 Classification Report](https://github.com/user-attachments/assets/2f3cb863-fd26-4232-846d-20b65ef4aee2)

![InceptionV3 Confusion Matrix](https://github.com/user-attachments/assets/edad0cb1-6e64-4fd8-8084-3c89bade68a3)

![InceptionV3 Roc Auc](https://github.com/user-attachments/assets/8497cb68-c467-4225-80c6-ef00676f7675)

*EfficientNetB0*

![EfficientNetB0 Classification Report](https://github.com/user-attachments/assets/7e99fd5f-7feb-4b57-ba74-abaf2a9df643)

![EfficientNetB0 Confusion Matrix](https://github.com/user-attachments/assets/f3bb7c36-42ac-4cf7-b34f-21e88d0ef00a)

![EfficientNetB0 Roc Auc](https://github.com/user-attachments/assets/181b75bf-e636-49c8-bff3-09af2684e62c)

**ROC Curve and AUC**

The Area Under the Curve (AUC) was calculated for each model to quantify their ability to distinguish between parasitized and uninfected cells. EfficientNetB0 showed the highest AUC score of 0.95, followed closely by Manual CNN and ResNet50.

## Performance Comparison

After analyzing the results, EfficientNetB0 emerged as the best model based on its balance of accuracy, computational efficiency, and validation performance. Although the Manual CNN also achieved comparable accuracy, EfficientNetB0's pre-trained nature and optimized architecture made it more suitable for deployment in real-world settings, especially in resource-limited environments.

![Compare CNN Model Performance](https://github.com/user-attachments/assets/3fed2799-a88c-45db-9d30-3b1a30f9c53b)

## Conclusion

This project successfully demonstrated the power of CNN models in automating the detection of malaria in blood smear images. By comparing five different models, EfficientNetB0 was identified as the most promising model for practical use, due to its accuracy and efficient design. Future work could involve deploying this model in mobile applications to further improve the accessibility and efficiency of malaria diagnosis in remote areas.

## Recommendations

* Deployment of the selected model (EfficientNetB0) in mobile or embedded systems for real-time malaria detection.
* Expanding the dataset to include more diverse blood smear images from different geographical regions to improve generalization.
* Investigation into further optimization techniques such as model pruning and quantization to reduce the model's size and computational requirements for deployment in low-resource settings.
