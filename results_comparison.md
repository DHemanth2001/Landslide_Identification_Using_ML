# Results Comparison
## How the Project Improved Over Time

This document explains the 5 different results (models) created during this project, and how each step was better than the last. This will help you answer questions about why you chose the final model.

### 1. Comparison Table

| Result | Method Used | Improvement from Previous | Reason | Performance (Accuracy) |
| --- | --- | --- | --- | --- |
| **Result 1 (r0)** | AlexNet (From Scratch) | (Baseline Model) | We needed a starting point to see how a basic model performs. | 78.54% |
| **Result 2 (r1)** | AlexNet (Pretrained) | Used "Transfer Learning" (Pre-trained weights). | The model was trained on 1.2 million images (ImageNet) before looking at our landslides. It learns better this way because our dataset is small. | 80.11% (+1.57%) |
| **Result 3 (r2)** | EfficientNet-B3 | Changed the model architecture to EfficientNet-B3. | EfficientNet is a more modern, efficient, and smarter architecture. It is much better at finding actual landslides (higher recall/F1-score). | 79.97% (More balanced) |
| **Result 4 (r3)** | EfficientNet-B3 (Calibrated) + Improved HMM | Added "Temperature Scaling" and made the HMM smarter (8 states instead of 6, and 42 combined symbols). | The previous model was too overconfident even when it was wrong. This step fixed the confidence scores. We also gave the HMM data about environmental triggers, making it smarter. | 79.97% (Better Confidence) |
| **Result 5 (r4)** | Ensemble + Optimal Threshold | Combined EfficientNet-B3 (60%) + AlexNet (40%), and tuned the decision threshold to 0.467. | Two models are better than one! By averaging their guesses, they cancel out each other's mistakes. Changing the threshold to 0.467 gave the best balance between False Positives and False Negatives. | 82.40% (+2.43%) |

---

### 2. Simple Explanation of Each Step

#### **Result 1: AlexNet (From Scratch)**
- **What we did:** We built a standard Artificial Neural Network called "AlexNet" and trained it entirely from scratch using only our 2,190 satellite images.
- **Why it was okay but not great:** The model got 78.54% accuracy, but deep learning models usually need hundreds of thousands of images to learn perfectly. 2,190 images were not enough for it to become an expert.

#### **Result 2: AlexNet (Pretrained / Transfer Learning)**
- **What we did:** Instead of starting from scratch, we took an AlexNet model that had already studied 1.2 million general images (like dogs, cats, cars, trees). We then "Transfer Learned" it to focus only on landslides.
- **Why it improved:** Because the model already knew how to see textures and shapes (like edges or brown patches), it jumped to 80.11% accuracy very quickly.

#### **Result 3: EfficientNet-B3**
- **What we did:** AlexNet is a very old architecture (from 2012). We upgraded the "brain" to a newer 2019 architecture called EfficientNet-B3.
- **Why it improved:** Even though the accuracy looked similar (79.97%), the **Recall** improved a lot. Recall means: *Out of all the real landslides in the world, how many did we successfully find?* Missing a landslide is dangerous, so this was a great improvement.

#### **Result 4: Calibrating Confidence & Improving the Future Predictor (HMM)**
- **What we did:** The model from Result 3 was acting like a cocky student—guessing wrong but being 99% confident about it. We applied a math trick called "Temperature Scaling" to make the model honest about its confidence levels. We also upgraded Phase 2 (the HMM predictor) by giving it environmental triggers (like Heavy Rain) to help it predict the future better.
- **Why it improved:** It made the system more trustworthy for real-world disaster management teams.

#### **Result 5: The Final Model (Ensemble)**
- **What we did:** Instead of throwing away AlexNet, we combined EfficientNet-B3 and AlexNet together into a team (called an Ensemble). We also tweaked the "Decision Threshold". Before, it needed 50% confidence to say "Landslide". We found that dropping it to 46.7% caught more real landslides without increasing false alarms too much.
- **Why it improved:** When one model was confused, the other model usually knew the right answer. Together, they achieved the highest accuracy yet (82.40%). This is the final and best version of the Phase 1 pipeline!
