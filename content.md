Landslide Identification Using Machine Learning
Project Content and Documentation


1. WHY THIS PROJECT WAS CHOSEN

Landslides are one of the most destructive and frequently occurring natural disasters in the world. They cause thousands of deaths every year, destroy infrastructure, displace communities, and inflict billions of dollars in economic loss. Countries like India, Nepal, China, the Philippines, and parts of South America are particularly vulnerable because of their mountainous terrain, heavy monsoon rainfall, and rapid urban expansion into unstable slopes.

Despite the severity of the problem, most traditional landslide identification systems rely on manual field surveys, which are slow, expensive, and impossible to scale across large geographic areas. Satellite imagery is widely available but analyzing it manually by experts is time-consuming and not practical for real-time monitoring or early warning systems.

This project was chosen to address a critical real-world problem using modern machine learning techniques. The goal is to build an automated system that can not only detect whether a satellite image contains a landslide but also classify what type of landslide it is and predict when and where the next landslide is likely to occur. This makes the system useful not just for detection after an event but also for prevention and preparedness before one happens.

The problem is also academically significant because it requires combining two fundamentally different types of machine learning: deep learning on visual data (images) and probabilistic sequential modeling on historical event data (time series). This makes the project more challenging and more comprehensive than a standard image classification task.


2. HOW THIS PROJECT IS DIFFERENT FROM EXISTING LANDSLIDE IDENTIFICATION SYSTEMS

Several landslide identification systems already exist, but they each have significant limitations. This project addresses those limitations directly.

Most existing systems use only remote sensing indices such as NDVI (Normalized Difference Vegetation Index) or change detection algorithms to flag potential landslide zones. These methods detect changes in vegetation or surface texture but cannot classify what type of landslide occurred or predict future events.

Some systems use Support Vector Machines or Random Forests trained on terrain features like slope angle, elevation, and land cover. While effective for susceptibility mapping, these models do not process raw image data and cannot generalize across different geographic regions without manual feature engineering.

A few recent deep learning systems apply U-Net or ResNet architectures for pixel-level segmentation of landslides in satellite images. These models are powerful but require very large annotated datasets and are not connected to any temporal prediction component.

None of the commonly available landslide identification systems combine image classification with a probabilistic future prediction model. This project is different in the following ways:

It uses a two-phase architecture where Phase 1 handles image-based detection and Phase 2 handles type classification and future prediction independently. Each phase can be improved or replaced without breaking the other.

Phase 2 is trained on 9,471 real-world landslide events from the NASA Global Landslide Catalog spanning 141 countries and over 28 years, making the predictions based on actual historical patterns rather than synthetic assumptions.

The system produces not just a binary yes or no decision but a complete risk profile including the type of landslide, the probability of occurrence, the peak risk month for the region, historical fatality rates, and a multi-step forecast of what types of landslides are likely to happen next.

The AlexNet model was trained on the HR-GLDD dataset which contains high-resolution 128x128 pixel satellite patches with precise pixel-level masks, making it far more accurate than systems trained on coarse resolution imagery.


3. DATASET SOURCES AND COLLECTION

Phase 1 Dataset — HR-GLDD (High-Resolution Global Landslide Detection Dataset)

This dataset was obtained from Zenodo, a scientific data repository maintained by CERN, under DOI 7189381. It was created by researchers who manually annotated satellite images with pixel-level landslide masks.

The original dataset consists of numpy array files in the format (N, 128, 128, 4) representing N patches of 128 by 128 pixels with 4 spectral bands — Red, Green, Blue, and Near-Infrared (NIR). The label files are binary masks of shape (N, 128, 128, 1) where each pixel is labeled 1 for landslide and 0 for background.

Since this project uses a classification (not segmentation) approach, the numpy arrays were converted to JPEG images using a custom script. Patches where more than 5 percent of pixels were labeled as landslide were classified as landslide images. For the non-landslide class, 64x64 sub-crops were extracted from regions of the same patches where the mask was entirely zero, then resized to 128x128 pixels.

The final image counts after conversion are as follows. The training set has 616 landslide images and 1574 non-landslide images for a total of 2190. The validation set has 158 landslide images and 392 non-landslide images for a total of 550. The test set has 211 landslide images and 488 non-landslide images for a total of 699.

Phase 2 Dataset — NASA Global Landslide Catalog (GLC)

This dataset was downloaded from the NASA Open Data Portal. It contains 11,033 records of global landslide events recorded between 1970 and 2019. After filtering for required fields, 9,471 complete records were used for training.

Each record contains the event date, country name, landslide category (type), landslide trigger, landslide size, and fatality count. The landslide categories are mapped to 7 canonical types: Landslide, Debris Flow, Mudslide, Rockfall, Earth Flow, Translational Slide, and Complex Landslide. The dataset covers 141 countries, making it truly global.

This dataset was chosen specifically because it is publicly available, peer-reviewed, curated by NASA scientists, and contains enough temporal breadth and geographic diversity to train a meaningful HMM for sequential prediction. The records span 28 years of real landslide history, allowing the model to learn genuine temporal patterns of how landslide types follow each other over time in different regions.


4. PROJECT ARCHITECTURE

The system is designed as a pipeline with two sequential phases. When a satellite image is provided, it first passes through Phase 1. The output of Phase 1 determines whether Phase 2 is invoked.

Phase 1 — AlexNet Convolutional Neural Network

The input is a satellite or aerial image of any size, which is resized to 227 by 227 pixels. During training, data augmentation is applied including random horizontal flipping and color jitter to improve generalization. The image is normalized using ImageNet mean and standard deviation values.

The AlexNet architecture consists of five convolutional layers followed by three fully connected layers. The first convolutional layer uses 96 filters of size 11x11 with a stride of 4. The second uses 256 filters of size 5x5. The third, fourth, and fifth layers use 384, 384, and 256 filters respectively, all of size 3x3. Each convolutional block is followed by ReLU activation, and pooling layers are applied after layers 1, 2, and 5.

The feature maps from the final convolutional layer are passed through an adaptive average pooling layer to produce a 6x6 feature map, which is flattened to 9216 values. Three fully connected layers of sizes 4096, 4096, and 2 follow, with Dropout of 0.5 applied after each of the first two FC layers. The final output is a 2-class softmax probability giving the probability of landslide versus non-landslide.

The model is trained using the Adam optimizer with a learning rate of 0.0001 and Cross Entropy Loss. A StepLR learning rate scheduler reduces the learning rate every 10 epochs. Class imbalance (3:1 ratio of non-landslide to landslide) is handled using a WeightedRandomSampler during training so both classes are seen equally.

Phase 2 — CategoricalHMM (Hidden Markov Model)

Phase 2 takes the country name as context input. The historical event records from the NASA GLC are grouped by country and sorted chronologically to form observation sequences. Each observation is an integer representing one of the 7 landslide types.

The HMM has 4 hidden states, which through training learn to represent 4 distinct landslide regimes: Shallow Landslide regime, Deep Landslide regime, Debris Flow regime, and Rockfall regime. The transition matrix captures how likely it is to move from one regime to another. The emission matrix captures which types of landslides are most commonly observed in each regime.

Training uses the Baum-Welch algorithm (a form of Expectation-Maximization) which iteratively adjusts the start probabilities, transition matrix, and emission matrix to maximize the likelihood of the observed sequences. Training runs for 100 iterations on 9,442 observations across 112 country sequences.

For prediction, Viterbi decoding finds the most likely sequence of hidden states given the observed landslide history. The current state determines the most likely current landslide type. The transition matrix is then propagated forward N steps to forecast the most likely type at each future step. The occurrence probability is computed from the log-likelihood of the observation sequence, normalized using a sigmoid function to produce a value between 0 and 1.

Pipeline Integration

When the full pipeline is run, Phase 1 first classifies the image. If the confidence is above 50 percent and the prediction is landslide, Phase 2 is automatically triggered. The combined output includes the image label, confidence score, landslide type, occurrence probability, peak risk month, historical statistics for the given country, and a 3-step future forecast.


5. TRAINING TIME

Phase 1 — AlexNet Training Time

The AlexNet model is trained for 30 epochs with a batch size of 32. On a standard CPU without GPU acceleration, each epoch takes approximately 8 to 12 minutes, making the total training time around 4 to 6 hours. With a GPU (NVIDIA with CUDA), training time reduces to approximately 15 to 25 minutes total. The best checkpoint is saved when validation accuracy improves, so training can be stopped early if the accuracy plateaus.

Phase 2 — HMM Training Time

The HMM training on 9,442 observations with 100 Baum-Welch iterations takes approximately 10 to 30 seconds on a CPU. HMMs do not require GPU and are computationally lightweight. The training converges well before 100 iterations in most cases.

Total Time for Full Training from Scratch

On CPU only: approximately 4 to 6 hours.
On GPU: approximately 20 to 30 minutes.


6. METRICS CALCULATED AND WHAT THEY MEAN

Phase 1 — Image Classification Metrics

Accuracy is the overall percentage of images correctly classified as landslide or non-landslide. This project achieves 79.54 percent. While useful as a quick measure, accuracy alone can be misleading when class counts are unequal.

Precision measures out of all images the model predicted as landslide, what fraction were actually landslide. This project achieves 64.91 percent. Low precision means the model sometimes raises false alarms, predicting landslide when the image does not contain one.

Recall measures out of all actual landslide images, what fraction did the model correctly detect. This project achieves 70.14 percent. In disaster detection, recall is critically important — missing a real landslide (low recall) is far more dangerous than a false alarm.

F1-Score is the harmonic mean of precision and recall. It gives a single balanced score. This project achieves 67.43 percent. It is the most useful metric for imbalanced classification tasks like this one.

ROC-AUC (Receiver Operating Characteristic — Area Under Curve) measures the model's ability to rank positive examples above negative examples across all possible decision thresholds. This project achieves 86.66 percent. A value above 85 percent is considered good. This means the model has strong discriminative ability even if its threshold-specific precision and recall have room to improve.

Phase 2 — HMM Metrics

Log-Likelihood is the primary metric for HMM evaluation. It measures how well the trained model explains the observed sequences. The training log-likelihood for this project is -7314.96 across 9,442 observations. A higher (less negative) value means the model fits the data better. The per-observation log-likelihood is approximately -0.775, which indicates the model has learned meaningful patterns from the data.


7. CAN THE ACCURACY BE IMPROVED

Yes, there are several concrete ways to improve performance.

For Phase 1 (AlexNet), the most impactful improvement would be to use transfer learning with pretrained ImageNet weights. The current model is trained from scratch (randomly initialized weights). Starting from ImageNet pretrained weights would give the model a much better starting point, especially for the convolutional layers that learn general visual features. This alone could push accuracy to 85 to 90 percent.

Another improvement is to use a more powerful base architecture. ResNet-50, EfficientNet-B3, or Vision Transformers (ViT) are significantly more accurate than AlexNet on image classification tasks, especially with limited training data.

Increasing the dataset size would also help substantially. The current training set has only 2,190 images. Augmenting with additional publicly available landslide datasets such as the Bijie dataset, the Landslide4Sense dataset, or images from Google Earth Engine could bring this to 10,000 or more images.

Using all 4 spectral bands (including NIR) instead of just RGB would provide more discriminative information since landslide regions often show spectral signatures in the near-infrared channel that are not visible in standard RGB.

For Phase 2 (HMM), increasing the number of hidden states from 4 to 6 or 8 might capture more nuanced landslide regimes. Using a Gaussian HMM with additional continuous features (like landslide size, fatality count, and trigger type as multi-dimensional observations) rather than just the categorical type would produce richer predictions.


8. COMPARISON WITH OTHER MODELS AND WHY ALEXNET AND HMM WERE CHOSEN

Comparison with other image classification models for Phase 1:

AlexNet was the first deep CNN to win the ImageNet competition in 2012 and remains a strong baseline for binary image classification tasks. It has 60 million parameters, which is large enough to learn complex visual features but not so large that it overfits on a dataset of 2,190 images.

VGG-16 has 138 million parameters. It is more accurate than AlexNet but takes significantly longer to train and requires more memory. For the scale of this project and the available compute resources, AlexNet is the more practical choice.

ResNet-50 uses residual connections to train very deep networks (50 layers) without vanishing gradient problems. It outperforms AlexNet on most benchmarks. However it is designed for large datasets and requires transfer learning to perform well on small datasets. AlexNet is simpler to train from scratch.

EfficientNet scales depth, width, and resolution together and achieves state-of-the-art results with fewer parameters. It would be a natural next step to improve this project in future iterations.

AlexNet was chosen for this project because it is the classic benchmark CNN that is well-understood in academic settings, fast to train on CPU, appropriate for binary classification, and produces results that are easy to explain and compare.

Comparison with other sequential models for Phase 2:

LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) neural networks are capable of learning long-term dependencies in sequences and would likely produce better raw accuracy on type prediction. However they require labeled sequences with known types at each timestep and are much harder to interpret. They also produce point predictions rather than probability distributions, making them unsuitable for the occurrence probability calculation this project requires.

Random Forest or Gradient Boosting classifiers could predict the next landslide type given features from the previous event. However they treat each prediction independently and cannot model the temporal dynamics of how landslide regimes evolve across a region over time.

CategoricalHMM was chosen because it is specifically designed for sequential categorical observations. It provides a full probabilistic model of the underlying hidden states (landslide regimes) and their transitions. The Viterbi algorithm gives the most likely state sequence, and the forward algorithm gives exact occurrence probabilities. It is also highly interpretable — the transition matrix directly shows how likely each regime is to change to another, and the emission matrix shows which types each regime produces. For a project in disaster risk management, interpretability is critical.


9. POSSIBLE FURTHER IMPROVEMENTS

The following improvements can make this project significantly stronger as future work.

Using a Vision Transformer (ViT) for Phase 1 to replace AlexNet. Transformers have recently outperformed CNNs on many image tasks, especially when fine-tuned from large pretrained models.

Adding a segmentation output to Phase 1. Instead of just classifying the whole image, a U-Net could output a pixel-level mask showing exactly which parts of the image contain the landslide. This would be more useful for field teams.

Incorporating real-time satellite data feeds from sources like Sentinel-2 or Landsat through APIs. This would allow the system to automatically monitor new images and trigger alerts.

Building a web interface or mobile application where a user can upload a satellite image, enter their location, and get an instant prediction report.

Extending Phase 2 to include continuous features such as rainfall data, seismic activity, slope gradient, and soil moisture as additional HMM observations. This would allow predictions to be tied to specific environmental triggers.

Adding confidence calibration to Phase 1. The current softmax probabilities are not well-calibrated, meaning a 70 percent confidence score does not actually correspond to 70 percent empirical accuracy. Temperature scaling or Platt scaling can fix this.

Training on multi-temporal image pairs (before and after satellite imagery) using Siamese networks to detect change, which is one of the most reliable methods for identifying new landslides.


10. QUESTIONS A GUIDE OR EXAMINER MAY ASK

Q: Why did you choose AlexNet over VGG or ResNet?
A: AlexNet is the classic baseline CNN that is well-suited for binary classification with limited training data. It trains faster on CPU than VGG or ResNet, is simpler to explain, and still achieves competitive results. It was the natural starting point for this academic project. ResNet or EfficientNet would be logical next-step upgrades.

Q: Your accuracy is 79.54 percent. Why is it not higher?
A: There are three main reasons. First, the training dataset is small — only 2,190 images. Second, the model is trained from scratch without pretrained weights, which means the convolutional layers must learn all visual features from the limited data. Third, the non-landslide images were synthetically generated from sub-crops of the same HR-GLDD patches, which introduces some similarity between the two classes. Using pretrained weights and a larger diverse dataset would realistically push accuracy to 85 to 90 percent.

Q: Why is the ROC-AUC much higher than accuracy? What does that tell us?
A: ROC-AUC at 86.66 percent measures the model's discriminative ability across all classification thresholds, not just the default 50 percent threshold. The high ROC-AUC compared to accuracy tells us the model produces well-ranked probability scores — it assigns higher probabilities to actual landslide images — but the default threshold of 0.5 may not be optimal. Adjusting the threshold based on the desired balance between precision and recall could improve practical performance.

Q: Why is the HMM trained on country-level sequences? Why not individual events?
A: Landslide occurrence follows regional patterns driven by geology, climate, and topography. A country represents a coherent geographic unit with shared environmental drivers. Grouping by country and sorting chronologically creates meaningful temporal sequences that reflect how landslide activity evolves within a region. Individual events without geographic grouping would be too noisy and would not form coherent sequences for the HMM to learn from.

Q: What is Viterbi decoding and why is it used?
A: Viterbi decoding is a dynamic programming algorithm that finds the single most likely sequence of hidden states given the observed sequence. It is used in Phase 2 to determine which hidden state (landslide regime) best explains the current pattern of observed landslide types. This is more efficient and accurate than evaluating all possible state sequences, especially as sequence length grows.

Q: How does the future forecast work?
A: Starting from the current hidden state determined by Viterbi decoding, the state probability distribution is projected forward by repeatedly multiplying it by the transition matrix. At each future step, the most probable state is identified, and the emission matrix is used to find the most likely landslide type that state would produce. The probability reported is the joint probability of being in that state and observing that type.

Q: Can this system be used for real-time disaster warning?
A: Yes, with additional engineering. The Phase 1 model can process a new satellite image in under one second. Phase 2 runs in milliseconds. The bottleneck would be data ingestion — getting up-to-date satellite imagery in near real-time. With integration to platforms like Google Earth Engine or ESA Copernicus, the pipeline could realistically provide automated alerts within hours of a new satellite overpass.

Q: Why does the HMM only predict type and not exact timing of the next event?
A: The HMM as implemented models the sequence of event types but does not encode precise calendar time between events. This is a deliberate simplification. To predict timing, one would need to add a duration model, for example a Poisson process or survival model, to estimate the inter-event time distribution. The current historical rate statistics (events per year, peak month) serve as a proxy for timing.

Q: How do you handle the class imbalance in the image dataset?
A: The training set has approximately 3 non-landslide images for every 1 landslide image. This is handled using a WeightedRandomSampler in PyTorch, which oversamples the minority class (landslide) so that the model sees equal numbers of both classes in each epoch. Without this, the model would be biased toward predicting non-landslide for all inputs and achieve high accuracy but very low recall on landslide images.

Q: What is the PHASE1_THRESHOLD and why is it set to 0.5?
A: The threshold is the minimum confidence score from Phase 1 that triggers Phase 2. If AlexNet assigns more than 50 percent probability to the landslide class, the image is classified as landslide and Phase 2 runs. The value 0.5 is the standard decision boundary for binary classification. In practice, it could be lowered to increase recall (catch more events) at the cost of more false alarms, depending on the application.

Q: Why is the NASA GLC preferred over other landslide databases?
A: The NASA Global Landslide Catalog is publicly accessible, peer-reviewed, maintained by NASA scientists, and covers 141 countries over 28 years with 9,471 records. Other databases such as NOAA's or regional government databases are either restricted, cover smaller geographic areas, or have fewer temporal records. The diversity and size of the NASA GLC makes it ideal for training a globally applicable HMM.

Q: What are the limitations of this project?
A: The main limitations are: the image dataset is small which limits Phase 1 accuracy; the HMM uses only the landslide type as the observation variable, ignoring other potentially useful features like trigger and size; the future forecast probabilities converge quickly to steady-state values because the transition matrix has dominant diagonal entries; and the system has not been tested on live satellite imagery, only on the HR-GLDD benchmark dataset.
