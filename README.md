# Early-Stress-Detection
Early Stress Detection in Agriculture using RGB and multispectral imagery 
# ABSTRACT

This study addresses the critical challenge of early non-disease crop stress identification through a deep learning framework optimized for multispectral aerial imagery. Utilizing the Agriculture-Vision dataset, we propose a pipeline that categorizes field conditions into Healthy, Early Stress, and Severe Stress based on area-ratio thresholding and the presence of catastrophic planter skips. Our methodology employs a 6-channel input tensor-comprising RGB, Near-Infrared (NIR), and two derived vegetation indices (NDVI and GNDVI)-processed via a Late Fusion. We compare two distinct methodologies: a high-capacity architecture utilizing ImageNet-pretrained ResNet-18 backbones and a custom-designed dual-branch CNN. Our findings indicate that the ResNet-18 model achieves a superior overall accuracy of 68.74% and a Macro F1-score of 0.6696,though it exhibits a substantial ~21% generalization gap (92% training vs. 71% validation accuracy). In contrast, the custom CNN, while reaching a lower accuracy of 61.33%, demonstrates improved generalization with a narrower accuracy gap (75% training vs. 65% validation). This study highlights the trade-off between the superior feature extraction of transfer-learning-based models and the architectural simplicity required to mitigate overfitting in complex agricultural scene analysis.. While the architecture demonstrates high discriminative power for Healthy and Severe Stress classes, our analysis highlights the inherent complexity of the Early Stress transitional phase.


**LIST OF FIGURES**


| Figure No. | Figure Title | Page No. |
| --- | --- | --- |
| Fig 1 | Distribution of data across stress ratio | 3 |
| Fig 2 | Distribution of data into categorical classes | 3 |
| Fig 3(a),3(b),3(c) | Sample images for classes 1,2,3 | 5-6 |
| Fig 4 | Architecture of Late Fusion Custom CNN Model | 7 |
| Fig 5(a), 5(b) | Training and validation curve for both models | 9 |
| Fig 6(a),6(b) | Prediction results/Confusion Matrix | 10 |
| Fig 7(a),7(b) | Normalized Prediction results | 11 |


<!-- vi -->

**LIST OF TABLES**


| Table No. | Table Title | PageNo. |
| --- | --- | --- |
| Table 1 | Data set Distribution and Label Instance Statistics | 2 |
| Table 2 | Training Configuration | 8 |
| Table 3 | Results | 9 |


<!-- vii -->

# 1.INTRODUCTION

Sustainable agricultural productivity requires the incorporation of new approaches to imnprove crop production and reduce the negative impact of plant stressors like drought, nutrient deficiency, and disease. Despite decades of research in plant physiology and pathology,early and accurate detection of plant stress remains a major challenge in sustainable agriculture[1].

In precision agriculture, the ability to identify abiotic and environmental stressors before they cause irreversible yield loss is paramount. Traditional monitoring often fails to distinguish between structural field issues and early-stage physiological decline. This paper focuses on the detection of four primary stressors: nutrient deficiency, water stress, weed clusters, and planter skips.

There has been a significant increase in scientific literature in recent years focusing on detecting stress in plants using hyperspectral image analysis. Plant disease detection is a major activity in the management of crop plants in both agriculture and horticulture. In particular,detecting early onset of stress and diseases would be beneficial to farmers and growers as it would enable earlier interventions to help mitigate against crop loss and reduced crop quality. Hyperspectral imaging is a non-invasive process where the plants are scanned to collect high-resolution data. The core contribution of this work is arobust assessment pipeline that integrates multispectral data through two primary innovations. First, we implement an area-based stress labeling logic that translates pixel-level semantic masks into actionable categorical classifications. Second, we present a Late Fusion CNN architecture specifically designed to ingest synchronized RGB and multispectral stacks. By isolating structural patterns (RGB) from physiological signals (Spectral indices), the model achieves a more nuanced understanding of field health than single-stream architectures.

“The aim of this project is to design and evaluate an deep learning system that can detect early-stage crop stress (such as water, nutrient, or heat stress) before visible dlisease symptoms appear, using fused RGB and multispectral imagery under real-world field conditions.”

<!-- 1 -->

# 2.DATASET ANALYSIS AND PREPROCESSING

## 2.1 Dataset Overview (Agriculture-Vision)

The research utilizes the Agriculture-Vision dataset[2], featuring 512x512 resolution multispectral imagery. Data distribution and label instance counts across the Train, Validation, and Test splits are detailed in Table 1.

Table 1: Dataset Distribution and Label Instance Statistics


| Metric/Label | Train | Validation | Test | Total |
| --- | --- | --- | --- | --- |
| Image Count | 14,628 | 3,779 | 4,220 | 22,627 |
| Label Instance Counts |  |  |  |  |
| NutrientDeficiency | 8,379 | 1,821 | 1,867 | 12,067 |
| Weed Cluster | 20,524 | 4,089 | 5,585 | 30,198 |
| Water | 2,699 | 1,214 | 1,195 | 5,108 |
| Planter Skip | 2,946 | 1,725 | 1,603 | 6,274 |
| $Waterway*$ | 1,343 | 245 | 664 | 2,252 |


*Note: Waterway is classified as an EXCLUDEDLABEL in our pipeline as it represents a structural farm feature rather than an environmental stressor. Furthermore, the labels doubleplant, drydown, endrow, and stormdamage were ignored due to zero instances in the source dataset.

## 2.2 Area-Based Stress Labelling Logic

To convert semantic masks into categorical labels, we developed a logic based on the spatial extent of stress relative to the valid field area:

1. **Total Valid** **Field** **Area** : Defined as the intersection

of fieldbounds and fieldmask.

2. **Stressed** **Area** **:** Defined as the union of pixels

in nutrientdeficiency, weedcluster, and water masks.

3. **Stress Ratio Calculation :**

$$\text {Ratio}=\frac {\text {StressedArea}}{\text {TotalValidFieldArea}}$$

**4. Categorization ThresholIds :**Calculated $33.3^{\mathrm {rd}}$ and $66^{\text {th}}$ percentile of the calculated stress ratio across unlabelled data for uniform distribution.

O $\text {Healthy:}\leq 9.5\%$ 

O**Early S** $\text {Stress:}9.5\%<\text {Ratio}\leq 40.7\%$ 

O**Severe Stress** **:** $\text {Ratio>}40.7\%\mathrm {OR}$ the presence of planterskip.

<!-- 2 -->

<!-- Distribution of Stress Ratios 1600 Healthy threshold(9.5%) Severe threshold (40.7%) 1400 1200 1000 Tuno0 a6euI 800 600 400 200 0 0.0 0.2 0.4 0.6 0.8 Stress Ratio -->
![](https://web-api.textin.com/ocr_image/external/0f408b8631391708.jpg)

Fig 1:Distribution of data across stress ratio

The planterskip label acts as a binary catastrophic override; if a planter skip is detected within the tile, the image is automatically classified as Severe Stress, bypassing the pixel-ratio calculation entirely.

Class Distribution Across Splits

<!-- Overall Distribution 8211 8000 7954 7000 6462 6000 5000 1n80 4000 3000 2000 1000 0 Healthy Early Stress Severe Stress -->
![](https://web-api.textin.com/ocr_image/external/575a108ba71350ae.jpg)

<!-- Train (n=14628) 452 5063 5000 4103 4000 1un0 3000 2000 1000 w Earty Stress Severe Stress -->
![](https://web-api.textin.com/ocr_image/external/cb3d87d6e615f5c3.jpg)

<!-- Val (n=3779) 1426 1378 1200 1000 975 第n8 800 600 400 200 0 Health Early Stress Severe Stress -->
![](https://web-api.textin.com/ocr_image/external/cbe42238039829e9.jpg)

<!-- Test (n=4220) 1465 1400 1384 1371 1200 1000 count 800 600 400 200 0 Health Earty Sbess Severe Stress -->
![](https://web-api.textin.com/ocr_image/external/8d654684d1d8bc82.jpg)

Fig 2: Distribution of data into categorical classes

<!-- 3 -->

# 3. METHODOLOGY

## 3.1 Vegetation Indices (VI) Mathematical Intuition

To augment the physiological feature space, we derive two vegetation indices. Vegetation Index is a spectral transformation of two or more bands designed to enhance the contribution of vegetation properties and allow reliable spatial and temporal inter-comparisons of terrestrial photosynthetic activity and canopy structural variations. These indices exploit the "red edge" phenomenon-the sharp increase in reflectance of vegetation in the NIR spectrum compared to the absorption in the visible spectrum[4].

### ·NDVI (Normalized Difference Vegetation Index):

$$NDVI=\frac {NIR-R}{NIR+R+ε}$$

Intuition:NDVI measures plant vigor by contrasting chlorophyll absorption of Red light $(0.6-0.7μm)$  with the high scattering of NIR light $(0.7-1.1μm)$  by leaf mesophyll.It serves as a proxy for biomass and overall photosynthetic capacity[3].

### ·GNDVI(GreenNormalized Difference Vegetation Index):

$$GNDVI=\frac {NIR-G}{NIR+G+ε}$$

Intuition: By replacing the Red band with the Green band, GNDVI offers higher sensitivity to chlorophyll concentration variations. While the Red band often saturates in dense canopies, Green reflectance remains sensitive to chlorophyll changes in later growth stages.

Note: e is maintained at 1e-7 for numerical stability.

<!-- 4 -->

<!-- Sample Images: RGB | NIR | NDVI | GNDVI (per class) -->

<!-- RGB-Healthy NIR NDVI GNDVI RGB-Healthy NR NDVI GNDVI RGB-Healthy NR NDVI GNDVI -->
![](https://web-api.textin.com/ocr_image/external/2f9dac6b49b10a55.jpg)

Fig 3(a): Sample images for class 1- Healthy

<!-- RGB-Early Stress NIR NDVI GNDVI RGB-Early Stress NIR NDVI GNDVI RGB-Early Stress NIR NDVI GNDVI -->
![](https://web-api.textin.com/ocr_image/external/d4f5b8bed209e44d.jpg)

Fig 3(b) : Sample images for class 2- Early Stress

<!-- 5 -->

<!-- RGB-Severe Stress NIR NDVI GNDVI RGB-Severe Stress NIR NDVI GNDVI RGB-Severe Stress NIR NDVI GNDVI -->
![](https://web-api.textin.com/ocr_image/external/bb4ea4e3cd99f04b.jpg)

Fig 3(c) : Sample images for class 3-Sever Stress

## 3.2 PROPOSED MODEL ARCHITECTURES

### 3.2.1 Late Fusion Dual-ResNet18 Architecture

The LateFusionCNN processes a 6-channel input tensor organized into two distinct streams:

**·RGB** **Branch:** A ResNet18 backbone processes the standard 3-channel [R, GG,B] data to identify structural anomalies, such as geometric patterns in planter skips or visible canopy thinning.

**·Spectral** **Branch:** A parallel ResNet18 backbone processes a 3-channel spectral stack [NIR, NDVI, GNDVI]. This is a non-standard applicationofthe ResNet18 architecture, designed to extract physiological features from infrared reflectance and chlorophyll-sensitive indices[5]

**·Fusion** **and** **Classification:** The 512-dimensional global average pooled features from both branches are concatenated into a 1024-dimensional vector. This fused representation passes through a three-stage fully connected head:

$$FC(1024\rightarrow 256)\rightarrow FC(256\rightarrow 128)\rightarrow FC(128\rightarrow 3)$$

<!-- 6 -->

### 3.2.2 Late Fusion Custom CNN Model

The proposed architecture, named **LateFusionCNN,** is a dual-branch Convolutional Neural Network (CNN) designed for agricultural stress detection from multimodal imagery. It operates on the principle of late fusion, where features are extracted from different data modalities (structural and physiological) independently and combined just before classification.

This approach allows the model to learn specialized spatial features relevant to each domain before fusingthem for a final holistic decision[6]

It consists of three main components:

1. **RGB** **Branch:** A CNN for visible feature extraction.

2. **Spectral Branch:** A CNN for physiological feature extraction.

3. **Fusion** **Classifier:** A multi-layer perceptron (MLP) that combines features and outputs the final classification logits.

**·Feature** **Extraction:** Each branch contains 4 convolutional blocks (Conv2d →BatchNorm2d→ ReLU→MaxPool2d), with filter counts increasing as $32\rightarrow 64\rightarrow 128\rightarrow 256.$ 

**·Fusion** **Arithmetic:**We utilize AdaptiveAvgPool2d to reduce spatial dimensions before flattening. The resulting feature length per branch is

$256\text {filters}\times 4\times 4$ spatial si $\text {size}=4096$ 

Upon concatenation, the8192-d vector is passed through

$$FC(8192\rightarrow 512)\rightarrow FC(512\rightarrow 128)\rightarrow FC(128\rightarrow 3).$$

<!-- Input:(B,6,H,W) RGB Image Spectral Image (R,G,B) (NIR,NDVI,GNDVI) RGB Branch Spectral Branch Input: RGB Image (B,3,H,W) Input: Spectral Image (NIR, NDVI, GNDVI) (B, 3, H,W) Output:(B,32,H/2,W/2) Conv Block 1 Conv Block 1 Output: (B,32,H/2,W./2) Conv2d :3.L, ReLU+X Pool Conv2d-324,ReLU+X Pool Output: (B, 64,H/4,W/4) Conv Block 2 Conv Block 2 Output:(B,64,H/4,W1/4) Conv2d =9.6. ReLU +X Pool Conv2d-604,ReLU+X Pool Output: (B,128,H/8,W/8) Conv Block 3 Conv Block 3 Output: (B,128,H/8,W/8) Conv2d 124,ReLU+X Pool Conv2d 128,ReLU+X Pool Output: (B,256,4,4) Conv Block 4 Conv Block 4 Output: (B,256,4,4) AdaptiveAvgPool2d(4x4) Concatenate AdaptiveAvgPool2d(4) Output:(B,256,4,4) Feature Fusion Output:(B,256,4,4) Flatten Output:(B,512) (B,8192 Features) Output:(B,512) Flatten Fully Connected Layer 512. BatchNormld(512) Dropout(0.5) Output:Healthy |Early Stress|Severe Stress Late Fusion CNN for Agricultural Stress Detection -->
![](https://web-api.textin.com/ocr_image/external/da8aa9d19910ef23.jpg)

Fig 4: Architecture of Late Fusion Custom CNN Model

<!-- 7 -->

# 4.EXPERIMENTAL SETUP

## 4.1 Training Hyperparameters

Both models utilized Adam optimization and class-weighted CrossEntropyLoss to account for the imbalanced distribution of healthy and stressed samples.

Table 22: Training Configuration


| Parameter | Dual-ResNet18 | Custom CNN |
| --- | --- | --- |
| Learning Rate | $1\times 10^{-4}$ | $1\times 10^{-3}$ |
| Weight Decay | $1\times 10^{-5}$ | $1\times 10^{-4}$ |
| LR Scheduler | ReduceLROnPlateau<br>(Factor: 0.5,Patience:3) | ReduceLROnPlateau |
| Input Resolution | $224\times 224$ | $224\times 224$ |
| Subsample Ratio | 1.0(Full Data) | 0.5(Half Data) |
| Early Stopping<br>Batch Size | 15 Epochs<br>32 | 10 Epochs<br>32 |


## 4.2 Data Augmentation

To enhance model robustness against varying aerial orientations, we applied random horizontal flips $(\mathrm {p}=0.5)$ ,random vertical flips $(p=0.5)$ ,and random rotations up to 15 degrees during the training phase.

<!-- 8 -->

# 5. RESULTS AND DISCUSSION

## 5.1 Performance Metrics

The custom CNN model outperformed the ResNet-18 across all primary metrics on the test set.

Table 3: Results


| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| --- | --- | --- | --- | --- |
| ResNet-18 | 68.74% | 0.6666 | 0.6835 | 0.6696 |
| Custom CNN | 75.24% | 0.6515 | 0.6575 | 0.6537 |


<!-- Training & Validation Loss Train Loss 0.9 Val Loss 0.8 0.7 S591 0.6 0.5 0.4 0.3 2.5 5.0 7.5 10.0 12.5 15.0 17.5 Epoch -->
![](https://web-api.textin.com/ocr_image/external/c8b699e8b3d181d4.jpg)

<!-- Training & Validation Accuracy Train Acc 85 Val Acc 80 75 (8)AeanC:y 70 65 60 55 2.5 5.0 7.5 10.0 12.5 15.0 17.5 Epoch -->
![](https://web-api.textin.com/ocr_image/external/48f0d7467a08a6f3.jpg)

Fig 5(a): Training and validation curve for ResNet-18

### ResNet-18 Max Accuracy -Epoch 29/50 (1052.6s) | LR: 1.25e-05

Train-Loss:0.2121 Acc:**91.73%**

Val-Loss: 0.9339 Acc:70.23%

Val-Precision: 0.6781 Recall: 0.6756 F1:0.6768

<!-- Training & Validation Loss Train Loss 1.1 Val Loss 1.0 0.9 Ss01 0.8 0.7 0.6 0 10 20 30 40 50 Epoch -->
![](https://web-api.textin.com/ocr_image/external/853160807d1c5f94.jpg)

<!-- Training & Validation Accuracy 75 Train Acc Val Acc 70 9 65 AeJnouy 60 55 50 0 10 20 30 40 50 Epoch -->
![](https://web-api.textin.com/ocr_image/external/c7286bd20063390c.jpg)

Fig 5(b):Training and validation curve for custom CNN

**CNN** **Max** **Accuray-Epoch** 48/50 (374.6s) | LR: 1.56e-05

Train - Loss: 0.5585 Acc: **75.24%**

Val - Loss: 0.7977 Acc: 65.86%

Val -Precision: 0.6515 Recall: 0.6575 F1:0.6537

<!-- 9 -->

## 5.2 Overfitting and Generalization Analysis

A detailed analysis of training dynamics reveals a significant divergence in generalization capabilities. Despite the use of a ReduceLROnPlateau scheduler and weight decay, the ResNet-18 model exhibited a ~21% gap between its training accuracy (~92%) and validation accuracy (~71%). This suggests that while the pretrained features provide a high performance floor, the model's capacity is high enough to begin memorizing domain-specific noise in the training set.

The Custom CNN, achieved a much tighter generalization gap (~10%). This indicates that shallower architectures may provide more stable performance in agricultural contexts where data labels are derived from pixel-area ratios rather than clear categorical boundaries.

## 5.3 Confusion Matrix Analysis

Analysis of normalized confusion matrices shows that ResNet-18 excels at polar classification, identifying "Healthy" samples at 81.17% and "Severe Stress" at 75.19%. The custom CNN showed lower sensitivity in these areas (69.44% and 63.34%,respectively).

Notably, both models struggled with "Early Stress," which was frequently misclassified as either Healthy or Severe (approximately 25% error rate in each direction). This reflects the inherent ambiguity of the "mniddle-class" in threshold-based labelling; a field with a 9.6% stress ratio is classified as Early Stress but is visually nearly identical to a Healthy field at 9.4%.

<!-- Confusion Matrix-Test Set -900 AeeH 961 299 124 -800 -700 -600 1ae1onL SSeIS NUe 356 699 316 -500 2n8 400 S3ens eJen8s -300 139 398 928 -200 Healthy Early Stress Severe Stress Predicted Label -->
![](https://web-api.textin.com/ocr_image/external/0d30996ac83e552d.jpg)

Fig 6: (a) Prediction results for CNN

<!-- Confusion Matrix-Test Set 1400 AIeeH 1496 251 96 1200 -1000 1ae1enL SecS Ne 367 702 373 800 1n8 -600 -400 SSeS aas 47 185 703 -200 Healthy Early Stress Severe Stress Predicted Label -->
![](https://web-api.textin.com/ocr_image/external/fbb33ff6c0f06969.jpg)

(b) Prediction results for Resnet-18

<!-- 10 -->

<!-- Normalized Confusion Matrix-Test Set AMeH 69.44% 21.60% 8.96% -0.6 -0.5 1ode1on4 SHeS Ae山 25.97% 50.98% 23.05% 0.4 UofAodor 0.3 SsaIDS PUanas 9.49% 27.17% 63.34% -0.2 0.1 Healthy Early Stress Severe Stress Predicted Label -->
![](https://web-api.textin.com/ocr_image/external/0bb7fb87e11ea258.jpg)

Fig 7: (a) Normalized Prediction results for CNN

<!-- Normalized Confusion Matrix-Test Set -0.8 AeeH 81.17% 13.62% 5.21% -0.7 -0.6 0.5 ode1onL SEeIS NUe山 25.45% 48.68% 25.87% 0.4 UoT0dod -0.3 S3eCS elaves 5.03% 19.79% 75.19% -0.2 -0.1 Healthy Early Stress Severe Stress Predicted Label -->
![](https://web-api.textin.com/ocr_image/external/712c791f62a09bbf.jpg)

(b) Normalized Prediction results for Resnet-18

The confusionmatrixtestnormalized provides a granular view of the model's performance:

**·Healthy** **Class(81.17%)**:The model effectively identifies healthy vegetation, benefiting from the distinct spectral signature of high NIR and low visible reflectance.

**·** **Severe** **Stress (75.19%)**: Strong performance here is likely due to the binary clarity of the planterskip override and the high pixel-ratio of stress masks, which create significant structural signatures.

**·Early** **Stress** **(48.68%)**: This class exhibits significant confusion, with 25.45% of samples misclassified as Healthy and 25.87% as Severe.

This performance gap suggests that the area-based ratio logic (9.5% to 40.7%) defines a transitional "grey zone." In this range, the visual and spectral features of stress-such as initial nutrient chlorosis or sparse weed clusters-are often indistinguishable from natural field variance or the early onset of severe conditions.

<!-- 11 -->

# 6. CONCLUSION AND FUTURE WORK

## 6.1 Conclusion

This project successfully designed and evaluated a Late Fusion Dual-Branch Convolutional Neural Network (CNN) for the early detection of crop stress using the Agriculture-Vision (data2019miniscale) dataset. By processing structural anomalies through an RGB branch and physiological anomalies through a spectral branch (utilizing NIR, NDVI, and GNDVI vegetation indices), the model effectively leverages multimodal data to identify non-disease stress factors.

This study concludes that deep, pretrained late-fusion architectures like Dual-ResNet18 are superior for extracting meaningful features from multispectral agricultural imagery,achieving a peak accuracy of 68.74%. However, the identified 21% generalization gap underscores the need for more aggressive regularization or broader datasets to fully leverage such high-capacity models. While custom CNN architectures offer lowver peak accuracy, their robust generalization and lower computational requirements make them viable for real-time edge applications. Future research should focus on refining the threshold-based labelling to reduce middle-class ambiguity and exploring more sophisticated fusion mechanisms beyond simple concatenation.

To adapt the semantic segmentation masks for image-level classification, an area-based thresholding metric was engineered, categorizing field images into "Healthy" $(\leq 9.5\%$ stress ratio), "Early Stress" (9.5% - 40.7%), and "Severe Stress" (&gt; 40.7%), while auto-assigning catastrophic labels like planterskip to severe stress. The best-performing fusion model, utilizing pretrained ResNet18 backbones, achieved an overall accuracy of **68.74%** and a Macro F1-Score of **0.6696.** While the model demonstrates the strong predictive capability of multispectral vegetation indices in agricultural monitoring, the confusion matrices indicate noticeable overlap between adjacent classes (e.g., Early Stress and Severe Stress), highlighting the limitations of rigid area-based thresholding and leaving room for pipeline optimization.

## 6.2 Challenges and Solutions in Future Implementations

While the proposed pipeline establishes a strong foundation, several challenges were observed during training and classification that can be addressed in future iterations to improve accuracy and reduce loss:

**· Suboptimal Class Distribution via Thresholding:** The current preprocessing pipeline creates discrete classes based on continuous stress-area ratios. This introduces rigid boundaries where an image with a 9.6% stress ratio is treated entirely differently than one with a 9.4% ratio, leading to misclassifications.

o Solution: Experiment with **shifting** **the** **thresholds** dynamically to establish clearer mathematical margins between "Healthy" and "Early Stress," which could yield better class accuracy, sharper feature distinction, and reduced cross-entropy loss.

**·** **Underutilization** **of Spatial** **Mask** **Data:** The dataset provides granular, pixel-level masks for highly specific conditions (e.g.,nutrientdeficiency,weedcluster,water).By collapsing these masks into a single "Stressed Area" ratio for whole-image classification, valuable spatial geometry and localized anomaly data are lost.

<!-- 12 -->

o Solution: Use the segmentation dataset **the way** it **was meant** to **be** **used-as** a direct semantic segmentation task. By mapping specific field labels directly to one of the three crop health categories at the pixel level, the stress threshold metric can be completely removed. This allows the model to learn the exact visual and spectral signatures of stress regions rather than a generalized area distribution.

**·Model Optimization and Hyperparameter Tuning:** The custom CNN branches and ResNet18 architectures occasionally plateaued during validation.

o Solution: **Improve** **the** **CNN** **architecture** by integrating spatial attention mechanisms to help the network focus specifically on the anomalous field patterns. Furthermore, adjusting the optimizer's parameters-such as **reducing** **the** **initial learning** **rate** (currently set at and in varying runs) or increasing the patience of the ReduceLROnPlateau scheduler-will allow the model to converge more smoothly without overshooting optimal weights during fine-tuning[7].

## 6.3 Future Scope

The long-term vision for this research extends beyond isolated image classification into real-time, scalable precision agriculture:

**·UAV** **Edge** **Deployment:** The late-fusion architecturecan be optimized and quantized for deployment on unmanned aerial vehicles (UAVs) equipped with multispectral cameras. This would allow drones to calculate NDVI and GNDVI on the fly,feeding the 6-channeI input into a lightweight CNN for real-time inference at the edge.

**·** **Expansion** to **Full-Scale** **Datasets**: Transitioning from the data2019miniscale subset to the full-scale Agriculture-Vision dataset will expose the model to higher variances in lighting,seasonal changes, and soil types, dramatically improving the model's robustness and generalization.

**·Precision** **Treatment** **Mapping:** By evolving the mode1 into a full semantic segmentation network as proposed above, the output could directly generate GPS-coordinated prescription maps. This would enable automated agricultural machinery to apply water, fertilizer, or herbicides exactly where needed (e.g.,isolated weedclusters or nutrientdeficiency zones), minimizing chemnical runoff and significantly reducing farming costs[8].

<!-- 13 -->

## Bibliography

1. Seralathan, P., Edward, A.S. Reinforcement learning based dynamic vegetation index formulation for rice crop stress detection using satellite and mobile imagery.Sci Rep 16,3447(2026).https://doi.org/10.1038/s41598-025-33386-9

2. Agriculture-Vision: A Large Aerial Image Database for Agricultural Pattern Analysis https://arxiv.org/abs/2001.01306

3. Lowe,A., Harrison, N. & French, A.P. Hyperspectral image analysis techniques for the detection and classification of the early onset of plant disease and stress. Plant Methods 13, 80 (2017). https://doi.org/10.1186/s13007-017-0233-z

4. Plant stress detection using multimodal imaging and machine learning: from leaf spectra to smartphone applications. Front. Plant Sci., 02 January 2026 Sec. Sustainable and Intelligent Phytoprotection Volume 16-2025 https://doi.org/10.3389/fpls.2025.1670593

5. A UAV-Based Multispectral and RGB Dataset for Multi-Stage Paddy Crop Monitoring in Indian Agricultural Fields https://doi.org/10.48550/arXiv.2601.01084

6. Machine Learning-Based Crop Stress Detection in Greenhouses https://doi.org/10.3390/plants1201005

7. A comprehensive review of crop stress detection: destructive, non-destructive, and ML-based approaches https://doi.org/10.3389/fpls.2025.1638675

8. Machine and Deep Learning: Artificial Intelligence Application in Biotic and Abiotic Stress MIanagement in Plants https://doi.org/10.31083/j.fbl2901020

<!-- 14 -->

