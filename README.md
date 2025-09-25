# Detection of Cardiac Arrest of New Born Babies Using Machine Learning

# Project Overview
This project focuses on the early detection of cardiac arrest in newborn babies within the Cardiac Intensive Care Unit (CICU) using machine learning models. Cardiac arrest in neonates is a rare but critical and life-threatening event. The primary goal of this project is to develop a predictive model that can alert healthcare professionals to at-risk infants before a critical event occurs, providing a more reliable and data-driven alternative to traditional diagnostic methods.

The project involves data preprocessing, model training, and evaluation using a dataset containing physiological parameters of newborns. The final predictive model is integrated into a simple web application for practical use.

# Problem Statement
The diagnosis of cardiac arrest in newborns often relies heavily on the clinical judgment of medical professionals. While expert judgment is invaluable, the subtle and rapid changes in a newborn's vital signs can be difficult to track in real-time, potentially delaying a life-saving response. This project addresses this challenge by applying machine learning to identify complex patterns in physiological data that are indicative of an impending cardiac event.

# Project Methodology
The project follows a structured methodology to ensure the reliability and robustness of the predictive models.

# Data Collection and Preparation:
A comprehensive dataset containing various physiological parameters of newborns, such as heart rate, oxygen saturation levels, blood pressure, and ECG results, was collected. This data forms the foundation for our predictive analysis.

# Data Preprocessing:
The raw data was cleaned and preprocessed to handle missing values, normalize features, and remove noise. This is a crucial step to ensure the data is in a suitable format for the machine learning algorithms.

# Model Training and Validation:
The preprocessed data was split into training and testing sets. The models were trained on the training data and validated using cross-validation techniques to prevent overfitting and ensure the models generalize well to new, unseen data.

<img width="1519" height="521" alt="Screenshot 2025-09-26 003231" src="https://github.com/user-attachments/assets/d12cb400-723d-460f-92b2-224346f5df67" />


# Machine Learning Models (Implemented From Scratch)
A core aspect of this project was the implementation of the core machine learning algorithms from scratch. This approach provided a deep understanding of their underlying mechanics, from fundamental logic to mathematical principles.

# 1. Decision Tree Classifier
A Decision Tree is a tree-like model of decisions and their possible consequences. In this project, it was built from the ground up by recursively splitting the dataset based on a chosen feature.

How it Works: The algorithm finds the best feature to split the data at each node by calculating metrics like Gini impurity or entropy. The goal is to create homogeneous groups (nodes) where all data points belong to the same class (cardiac arrest or no cardiac arrest).

From Scratch: We implemented the logic for finding the optimal split point, recursively building the tree, and making predictions by traversing the tree from the root to a leaf node.

<img width="632" height="438" alt="Screenshot 2025-09-26 003630" src="https://github.com/user-attachments/assets/26682b37-01e5-4ea0-a820-03e347c8b704" />

# 2. Logistic Regression
While called 'regression,' this model is a powerful tool for binary classification. It was chosen to determine the probability of a newborn experiencing cardiac arrest.

How it Works: The model uses a sigmoid function to transform a linear combination of input features into a probability score between 0 and 1. We then set a threshold (e.g., 0.5) to classify the output.

From Scratch: We implemented the sigmoid function and used gradient descent to iteratively update the model's weights and bias, minimizing the difference between the predicted probabilities and the actual outcomes.

<img width="1024" height="1024" alt="Gemini_Generated_Image_ebyrjwebyrjwebyr" src="https://github.com/user-attachments/assets/85d8811e-fe87-40cc-8464-8a74cf8c03f3" />

# 3. Support Vector Machine (SVM)
SVM is a highly effective model for classification. Its goal is to find the optimal hyperplane that separates the data points into different classes with the widest margin.

How it Works: For linearly separable data, the model finds the hyperplane that maximizes the margin between the two classes. For non-linear data, it uses the "kernel trick" to map the data into a higher-dimensional space where a linear hyperplane can be found.

From Scratch: Our implementation focused on finding the optimal hyperplane by solving an optimization problem. We also explored different kernels to handle complex, non-linear relationships in the physiological data.

<img width="1024" height="1024" alt="Gemini_Generated_Image_4g2icl4g2icl4g2i" src="https://github.com/user-attachments/assets/70e596a8-d93d-45e1-89cb-69aec07823c6" />

# Results & Conclusion
The trained models, particularly the Decision Tree and SVM, demonstrated high accuracy in predicting potential cardiac arrest scenarios. The performance was validated through cross-validation, confirming the robustness and reliability of the models. The project successfully demonstrates the potential of machine learning as a predictive tool to assist healthcare professionals in a critical care environment.

# Future Scope
As neonatal mortality remains a significant global concern, the potential for this project is immense. Future work could include:

Real-time Prediction: Integrating the model with real-time monitoring systems (e.g., wearable devices, IoT sensors) to provide instant alerts.

Broader Datasets: Expanding the dataset to include more diverse parameters and larger patient populations to improve model generalization.

Model Deployment: Developing a more sophisticated, production-ready application for use in clinical settings.

# How to Run the Project
This project includes a simple web application built with Django and Python. To run it, follow these steps:

Clone the repository:
git clone <repository-link>

Navigate to the project directory:
cd <project-directory>

Install the required dependencies:
pip install -r requirements.txt

Run the Django server:
python manage.py runserver
The application will be accessible at http://127.0.0.1:8000
