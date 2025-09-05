#### Name - Yashraj Ramdas Chavan
#### Roll no. - DA25M031
#### Assignment 2 - Data Analytics Lab ( Da5401) JUL-NOV 2025 
# DA5401: A2 Dimensionality Reduction, Visualization, and Classification Performance
### _OBJECTIVE: To apply concepts of vector spaces, dimensionality reduction, and feature engineering to a real-world classification problem. Use Principal Component Analysis (PCA) to reduce the data's dimensionality and then evaluate how this reduction affects the performance of a Logistic Regression classifier_
# 📂 Repository Structure
.

├── 📄 DALabA2.ipynb

├── 🍄 mushrooms.csv

└── 📖 README.md
# Mushroom Classification: A Study in Dimensionality Reduction
# Part A: Exploratory Data Analysis (EDA) & Preprocessing
## Chapter 1: Data Acquisition and Some Preprocessing
The analysis commences with the Mushroom Dataset, a collection characterized by numerous attributes and entirely categorical features. The primary objective is to classify mushrooms based on these attributes as either edible or poisonous. This dataset's inherent dimensionality presents an opportunity to explore dimensionality reduction techniques.
The initial steps involve loading the data and examining its basic structure to understand the variables and data types present, and performing some initial data cleaning.
1.  **Data Loading**: Load the mushroom dataset into a pandas DataFrame.
2.  **Initial Inspection**: Display the first few rows and review the data information (data types, non-null values) to get a foundational understanding of the dataset.
3.  **Data Cleaning**: Identify and remove columns with only one unique value, as these columns provide no discriminatory information.
## Chapter 2: Feature Engineering
Machine learning algorithms, particularly those that analyze variance and distance like PCA, require numerical data.
1.  **Feature-Target Separation**: Distinguish the features (`X`) used for prediction from the target variable (`y`) representing the mushroom class.
2.  **Target Encoding**: Convert the nominal target variable ('e' for edible, 'p' for poisonous) into a numerical format (0, 1).
3.  **One-Hot Encoding**: Apply One-Hot Encoding to the categorical feature set (`X`).
   
**Why One-Hot Encoding Prior to PCA?**
- PCA is a variance-based technique operating within vector spaces defined by numerical features. PCA Understands Only Numbers: It uses math (like measuring distances and variance), which only works on numbers. It can't understand text categories like 'red', 'green', or 'blue' on its own.
- Categorical variables lack inherent numerical magnitude or order. Directly assigning numerical labels would impose an artificial ordinal scale.
- One-Hot Encoding is the Solution: This technique translates categories into numbers without creating a fake ranking. It takes a single column like cap-color and creates a new, separate column for each color.
  Instead of one column with the word 'red', you get a column named is_red with a 1 (for yes) or 0 (for no).It does this for every color, so each category becomes its own independent, numerical feature.This allows PCA to analyze the categories mathematically, treating each one as a distinct concept without any artificial order.
  
## Chapter 3: Data Scaling - Standardization
Standardization is a critical preprocessing step for PCA. It involves transforming features to have a mean of 0 and a standard deviation of 1. This is necessary even for binary features.

1.  **Standardization Implementation**: Apply `StandardScaler` from scikit-learn to the one-hot encoded features.
   
**Importance of Standardization for One-Hot Encoded Data:**
- While one-hot encoded features are binary, the distribution and variance of these binary features can differ based on the frequency of the original categories in the dataset.
- PCA calculations are based on the covariance matrix, which is influenced by feature variances. Features with higher variances (e.g., representing categories that appear more frequently) would disproportionately affect the principal components.
- Standardization ensures that each feature contributes equally to the variance calculation by scaling them to a common range. This prevents features with higher frequencies from having an undue influence on the determined principal components, leading to a more robust PCA result that reflects the underlying data structure rather than the scale of specific features.
  
# Part B: Principal Component Analysis (PCA)
## Chapter 4: Applying Principal Component Analysis
With the data scaled, we proceed with Principal Component Analysis. PCA identifies a new set of orthogonal axes (principal components) ordered by the amount of variance they capture from the original data.

1.  **PCA Application**: Perform PCA on the standardized dataset without initially limiting the number of components. This provides the explained variance for all possible principal components.
   
## Chapter 5: The Scree Plot
The Scree Plot is an essential visualization for understanding how the total variance is distributed across the principal components and for guiding the selection of the number of components to retain.

1.  **Scree Plot Generation**: Create a plot displaying the explained variance ratio of each individual principal component and the cumulative explained variance ratio.
2.  **Optimal Component Determination**: Identify the minimum number of components required to explain a substantial proportion of the total variance, commonly set at 95%.
## Chapter 6: Visualizing Data in Reduced Dimensions
To gain a qualitative understanding of how the data distributes in the reduced PCA space, we create visualizations based on the principal components.
1.  **2D Projection Visualization**: Project the standardized data onto the first two principal components and generate a scatter plot, differentiating classes by color and marker style.
2.  **Multi-Dimensional Pair Plots**: Create a matrix of scatter plots (pair plot) for the top five principal components to examine relationships between different pairs of prominent components.
3.  **Separability Analysis**: Based on these visualizations, discuss the apparent linear separability of the two mushroom classes in the reduced feature space.
   
# Part C: Performance Evaluation with Logistic Regression
## Chapter 7: Performance Evaluation - Classification Model Comparison

To quantitatively evaluate the impact of PCA on classification performance, we train and evaluate Logistic Regression models on both the original, high-dimensional dataset and the PCA-transformed dataset. Logistic Regression is used as a simple, interpretable classifier to assess if the reduced feature set retains the necessary information for effective linear classification.

1.  **Data Splitting**: Partition the standardized dataset into training and testing subsets.
2.  **Baseline Model Training**: Train a Logistic Regression model using the full set of standardized features on the training data.
3.  **Baseline Model Evaluation**: Assess the baseline model's performance on the test set using standard metrics (accuracy, precision, recall, F1-score).
4.  **PCA Data Transformation**: Apply the previously fitted PCA transformation (using the optimal number of components determined in Chapter 5) to both the training and testing sets. It is critical to fit PCA only on the training data to prevent data leakage.
5.  **PCA Model Training**: Train a new Logistic Regression model using the PCA-transformed training data.
6.  **PCA Model Evaluation**: Evaluate the PCA-transformed model's performance on the transformed test data.
### Splitting Data for Model Training and Evaluation

We divide the standardized feature set and the corresponding target variable into distinct training (70%) and testing (30%) subsets. Stratified splitting is employed to ensure that the class distribution (edible/poisonous) is proportionally maintained in both the training and test sets.
### Training and Evaluating the Baseline Logistic Regression Model

A Logistic Regression model serves as our baseline, trained on the complete set of 116 standardized features. Its performance on the test set will provide a benchmark against which the PCA-based model can be compared.
### Training and Evaluating the PCA-Transformed Logistic Regression Model

We train a second Logistic Regression model, this time utilizing the PCA-transformed data, which comprises the optimal number of principal components (59) that collectively explain 95% of the variance. The PCA transformation is applied consistently to both the training and testing subsets after being fitted solely on the training data.
## Chapter 8: Comparative Analysis and Interpretation

This final stage involves comparing the performance metrics of the two trained models – the baseline model using full features and the PCA-transformed model. This comparison shows PCA's effectiveness in reducing dimensionality without compromising classification accuracy.

1.  **Performance Visualization**: Present a visual comparison of the number of features used by each model and their respective classification accuracies.
2.  **Performance Difference Analysis**: Analyze whether a significant difference in performance exists between the two models. Discuss potential reasons for the observed outcome, considering the trade-off between dimensionality reduction and information loss.
3.  **Impact of PCA**: Discuss whether PCA's inherent capability to handle feature collinearity and redundancy appeared to contribute to maintaining or improving performance.
## Final Chapter : The Conclusion

### The Efficiency of Dimensionality Reduction using PCA
The analysis of the Mushroom Dataset, guided by Principal Component Analysis, aimed to determine if a significant reduction in data dimensionality could be achieved while preserving the ability to accurately classify mushrooms. The findings provide a clear answer:

- **Dimensionality Reduction Achieved:** PCA successfully reduced the feature space from 116 dimensions (post-one-hot encoding) to **59 principal components**, while retaining 95% of the dataset's total variance. This represents a substantial simplification of the data representation.
- **Classification Performance Maintained:**
  - The baseline Logistic Regression model, trained on the full feature set, demonstrated high classification accuracy.
  - The Logistic Regression model trained on the PCA-transformed data achieved a nearly identical, also high, level of accuracy.
  - The difference in performance between the two models was negligible.
- **Successful Trade-off:** The outcome highlights a successful trade-off between dimensionality reduction and predictive power. The ability to achieve comparable high accuracy with less than half the number of features indicates that most of the original feature space contained redundant or less informative variance with respect to the classification task.
- **PCA's Contribution:** While not leading to a _higher_ accuracy in this instance (as the original data already allowed for near-perfect classification), PCA's process of identifying orthogonal components and consolidating variance effectively compressed the relevant information. This results in a more efficient and potentially more robust model by mitigating issues associated with high dimensionality and collinearity.
- **Logistic Regression's Insight:** Using Logistic Regression as a surrogate model provided valuable insight. Its high performance on the PCA-transformed data suggests that the principal components captured linearly separable patterns sufficient for accurate classification. This confirms that PCA preserved the essential discriminatory information in the dataset.

In summary, Principal Component Analysis proved to be a highly effective technique as it enabled a significant reduction in the complexity of the data representation, leading to a simpler model, without sacrificing the ability to accurately classify mushrooms. This underscores the practical value of PCA in preprocessing high-dimensional datasets for machine learning tasks.
