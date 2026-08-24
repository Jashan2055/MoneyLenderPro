# 💳 Micro-Lending Platform: End-to-End Credit Risk Assessment

An end-to-end micro-lending / credit-risk capstone that takes loan data from a MySQL database through data cleaning, feature engineering, exploratory analysis, machine-learning risk prediction, risk scoring, FastAPI integration, and a Streamlit user interface.

The final application predicts the probability that a loan will be a **bad loan** and converts that probability into a simple risk tier:

- **Low Risk**
- **Medium Risk**
- **High Risk**

The final model is a tuned **XGBoost classifier**.
> Built with a suspicious amount of Pandas, a few RAM-related breakdowns, and one very persistent XGBoost model. ☕

---

## 🚀 The Journey
The goal was not only to train a classifier, but to build a complete lending workflow:

This project started as:

> "Let's just train a credit-risk model."

It eventually became:

```text
2.26M rows
   ↓
"My RAM is dying."
   ↓
Chunk processing
   ↓
"Okay, let's engineer some features."
   ↓
Logistic Regression
   ↓
"ROC-AUC: 0.64... yeah, no."
   ↓
XGBoost
   ↓
"Much better."
   ↓
"Wait... are grade/sub-grade leaking information?"
   ↓
Remove grade features
   ↓
Hyperparameter tuning
   ↓
FastAPI
   ↓
Streamlit
   ↓
"Okay, NOW we're done."
```
And then:

> Spoiler: We survived. The model survived. The laptop... mostly survived. 💀



---
## 🎯 Key Features

- Chunk-based processing for 2.26M+ loan records
- Automated missing-value analysis and cleaning
- Credit-risk feature engineering
- XGBoost-based bad-loan probability prediction
- Risk-tier classification
- FastAPI prediction API
- Streamlit-based user interface
- SQL-based business insights

---
## Streamlit UI


<img width="897" height="595" alt="Image" src="https://github.com/user-attachments/assets/73a67f0d-a0f8-49ea-ba44-6462fd60a559" />

<img width="913" height="557" alt="Image" src="https://github.com/user-attachments/assets/53b72192-482e-472d-a47a-fadf530721f5" />

<img width="874" height="567" alt="Image" src="https://github.com/user-attachments/assets/cfc5c718-261d-4364-a812-aa91bb16432e" />

---



## 📊 Did It Actually Work?
### The interesting part

The first model wasn't exactly impressive.

```text
Logistic Regression
ROC-AUC: 0.6415
```
XGBoost changed the game:

```text
XGBoost
ROC-AUC: 0.7266
```
Removing the existing grade and sub_grade features gave us a slightly better model:
```text
Tuned XGBoost
ROC-AUC: 0.7299
```
Not a magical 99% accuracy model.

Just a reasonably useful credit-risk model that actually tells us something. 😌


| Model | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6415 | 0.2833 | 0.5448 | 0.3728 |
| XGBoost | 0.7266 | 0.3386 | 0.6408 | 0.4431 |
| XGBoost without Grade | 0.7298 | 0.3349 | 0.6631 | 0.4451 |
| **Tuned XGBoost** | **0.7299** | **0.3352** | **0.6643** | **0.4455** |

### 📈 Risk Tier Validation

| Risk Tier | Applicants | Actual Bad Rate |
|---|---:|---:|
| Low Risk | 68,880 | 6.17% |
| Medium Risk | 130,373 | 18.02% |
| High Risk | 62,025 | 39.90% |

The increasing bad-loan rate across the tiers shows that the model is able to separate applicants into progressively higher-risk groups.

---

##  🗃️ Dataset

- Original records: **~2.26M**
- Modeling records: **1,306,387**
- Good loans: **79.91%**
- Bad loans: **20.09%**

Because of the dataset size, large files are processed using **chunk-based processing** to avoid excessive memory usage.

---

## 🛠️ Tech Stack

**Data:** Python, Pandas, NumPy, MySQL  
**ML:** scikit-learn, XGBoost  
**API:** FastAPI, Uvicorn, Pydantic  
**UI:** Streamlit

---

## 📁 Project Structure

```text
├── database/
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── modeling/
│   ├── api/
│   └── app/
├── models/
├── data/
├── docs/
├── README.md
├── requirements.txt
└── .gitignore
```

---

# How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Microlending_capstone
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create `.env` in the project root:

```env
DB_HOST=localhost
DB_PORT=3306
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password
```

Never commit the `.env` file.

### 5. Start FastAPI

```bash
python -m uvicorn src.api.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

### 6. Start Streamlit

In another terminal:

```bash
streamlit run src/app/streamlit_app.py
```

Application:

```text
http://localhost:8501
```

---

## API

### Health

```http
GET /health
```

### Prediction

```http
POST /predict
```

The prediction endpoint receives applicant information and returns the predicted bad-loan probability and risk tier.

👉 [API Documentation](Documentation/API.md)

---

##  📚Documentation

Want to understand how the project was actually built?

### Project Walkthrough

Detailed explanation of the complete development process, including data processing, feature engineering, modeling decisions, problems encountered, and solutions.

👉 [Read the Project Walkthrough](Documentation/PROJECT_WALKTHROUGH.md)

### Architecture

👉 [Read the Architecture Documentation](Documentation/ARCHITECTURE.md)

### Model Card

Model configuration, evaluation, features, results, and limitations.

👉 [Read the Model Card](Documentation/MODEL_CARD.md)

### SQL & Business Analysis

👉 [Read the SQL Insights](Documentation/SQL_INSIGHTS.md)

---

## ⚠️  Limitations

This is a capstone credit-risk decision-support system and should not be treated as a production lending decision engine.

The model has useful predictive ability but is not perfect. Real-world deployment would require additional validation, calibration, fairness analysis, monitoring, and regulatory review.

---

## 🔮Future Improvements

- SHAP-based individual predictions
- Probability calibration
- Model monitoring and drift detection
- Fairness analysis
- Docker deployment
- Cloud deployment
- Automated retraining

---

## ✅Project Status

**Core project complete.**

The complete development process and technical decisions are documented in the [Project Walkthrough](Documentation/PROJECT_WALKTHROUGH.md).
