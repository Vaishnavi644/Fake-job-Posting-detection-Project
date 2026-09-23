# Fake Job Posting Detection

A Machine Learning project that classifies job postings as **Legitimate** or **Fraudulent** by analyzing textual and structured job-related data, built as part of the IBM SkillsBuild Data Analytics with AI Academic Internship Program (conducted by BharatCares in association with AICTE).

## Project Description

Online recruitment platforms are increasingly targeted by scammers who post fake job advertisements to deceive applicants, collect personal information, or commit financial fraud. This project builds an automated Machine Learning system that analyzes job posting text (title, company profile, description, requirements, benefits) along with structured attributes (employment type, telecommuting, presence of a company logo, etc.) to flag postings as fraudulent or genuine.

The workflow covers:
- Exploratory Data Analysis (missing values, class imbalance, feature distributions)
- Text cleaning and preprocessing (lowercasing, punctuation/stopword removal, lemmatization)
- Feature extraction using TF-IDF and CountVectorizer
- Training and comparison of four classification models: Logistic Regression, Naive Bayes, Decision Tree, and Random Forest
- Hyperparameter tuning using GridSearchCV
- Model evaluation using Accuracy, Precision, Recall, and F1-Score
- A functional prediction system for classifying unseen job postings
- Model interpretability analysis (identifying words most associated with fraud)

## Dataset

**Real / Fake Job Posting Prediction Dataset** (Kaggle)
Link: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

- 17,880 job postings, 18 columns
- Target variable: `fraudulent` (0 = Real, 1 = Fake)
- Combines textual fields (title, company_profile, description, requirements, benefits) with structured fields (employment_type, required_experience, telecommuting, has_company_logo, etc.)

## Technologies Used

- **Language:** Python 3
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Natural Language Processing:** NLTK (stopwords, WordNet Lemmatizer)
- **Machine Learning:** scikit-learn (TF-IDF/CountVectorizer, Logistic Regression, Naive Bayes, Decision Tree, Random Forest, GridSearchCV)
- **Environment:** Google Colab / Jupyter Notebook

## Project Files

| File | Description |
|---|---|
| `Vaishnavi_FakeJobPostingDetection.ipynb` | Complete project code (EDA, preprocessing, modeling, evaluation, prediction system) |
| `requirements.txt` | Python libraries required to run the notebook |
| `Vaishnavi_ProjectReport.docx` | Full project documentation and report |
| `README.md` | This file |
| `fake_job_postings.csv` | Dataset used (downloaded from Kaggle) |
| `best_fake_job_model.pkl` | Saved trained model (tuned Logistic Regression) |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer used with the model |

## Setup and Run Instructions

1. **Clone/download** this project folder, or open the notebook directly in Google Colab.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK resources** (run once, inside the notebook or a Python shell):
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('omw-1.4')
   ```

4. **Place the dataset** (`fake_job_postings.csv`) in the same directory as the notebook, or update the file path in the notebook accordingly.

5. **Run the notebook** cell by cell from top to bottom (or use "Run All" in Jupyter/Colab). This will:
   - Perform EDA and preprocessing
   - Train and evaluate all four models
   - Tune the final Logistic Regression model
   - Save the trained model (`best_fake_job_model.pkl`) and vectorizer (`tfidf_vectorizer.pkl`)

6. **Use the prediction system** to classify a new job posting:
   ```python
   import pickle

   model = pickle.load(open("best_fake_job_model.pkl", "rb"))
   tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

   def predict_job_posting(text):
       cleaned = clean_text(text)  # defined in the notebook
       vector = tfidf.transform([cleaned])
       prediction = model.predict(vector)[0]
       return "Real Job Posting" if prediction == 0 else "Fake Job Posting"

   print(predict_job_posting("We are looking for an experienced Python Developer..."))
   ```

## Key Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 97.31% | 1.000 | 0.445 | 0.616 |
| Naive Bayes | 96.11% | 0.681 | 0.370 | 0.479 |
| Decision Tree | 97.51% | 0.741 | 0.746 | 0.744 |
| Random Forest | 98.29% | 0.991 | 0.653 | 0.787 |
| **Tuned Logistic Regression (Final Model)** | **98.49%** | **0.984** | **0.699** | **0.818** |

The final tuned Logistic Regression model (optimized via GridSearchCV) was selected as the deployed model, achieving the best overall F1-Score while maintaining strong precision.

## Author

**Vaishnavi**
B.Tech Computer Science and Engineering
Government Women Institute of Technology (WIT), Dehradun
IBM SkillsBuild Data Analytics with AI Academic Internship Program (BharatCares × AICTE)
