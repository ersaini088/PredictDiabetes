# 🩺 Applying XAI Techniques to Diabetes Prediction Models

## 📘 Overview

This project explores how we can use **Explainable Artificial Intelligence (XAI)** techniques like **SHAP** and **LIME** to understand predictions made by machine learning models for **Diabetes prediction**.

We trained and compared three models:
- ✅ Logistic Regression
- ✅ Decision Tree Classifier
- ✅ Random Forest Classifier

Using SHAP and LIME, we analyzed how each model makes its predictions — both globally (feature importance) and locally (individual explanation).

---

## 📈 Evaluation Metrics

Each model is evaluated using the following metrics:

- **Accuracy** – Overall correctness of the model
- **Precision** – Out of all predicted positives, how many were actually positive?
- **Recall** – Out of all actual positives, how many did the model correctly predict?
- **F1 Score** – Harmonic mean of precision and recall (balances false positives/negatives)

---

## 🧪 What You’ll Learn

- How to train **Logistic Regression**, **Decision Tree**, and **Random Forest** classifiers
- How to use **SHAP** to understand feature contributions
- How to use **LIME** to explain individual predictions
- How to **compare SHAP and LIME** using:
  - Sparsity (how many features each method uses)
  - Stability (consistency between runs)
  - Fidelity (how well they imitate the original model)

---

## 🧠 Tools and Libraries

- Python 3.10+
- scikit-learn
- SHAP
- LIME
- NumPy & Pandas
- Matplotlib & Seaborn

---

## 📝 How to Run the Project

1. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap lime
```

2. Place the dataset `diabetes-dataset.csv` in the correct path

3. Run any of the scripts:

```bash
python PredictDiabetesDecisionTree.py
python PredictDiabetesLogisticRegression.py
python PredictDiabetesRandomForest.py
```

---

## 📊 Output Visuals

Each model will generate:
- A Feature Correlation Matrix
- Global Feature Importance plot
- SHAP Summary and Dependence Plots
- LIME HTML Explanation
- SHAP vs LIME comparison plots

---

## 📚 References & Useful Reading

These papers helped in understanding and implementing this project:

1. **IET Healthcare Technology Letters**  
   [Explainable AI for medical diagnosis](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/htl2.12039)

2. **ScienceDirect: Heliyon**  
   [XAI for early diabetes detection](https://www.sciencedirect.com/science/article/pii/S2405844024121433)

3. **Heliyon 2025**  
   [Hybrid Decision Trees for Health Monitoring](https://www.sciencedirect.com/science/article/pii/S2772442525000097)

4. **arXiv Preprint**  
   [Interpretable ML for Clinical Decision Support](https://arxiv.org/abs/2501.18071)

5. **Engineering Reports (Wiley)**  
   [Model-Agnostic Explanations](https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.13080)

6. **BMC Medical Informatics**  
   [Performance of XAI on diabetes classification](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-024-02810-x)

7. **MDPI - Information Journal**  
   [Overview of Explainability in AI](https://www.mdpi.com/2078-2489/16/1/7)

8. **ProQuest - Student Project**  
   [Case study on AI in Healthcare](https://www.proquest.com/openview/a5802c3d4d5a72660ad9d8e380165b92/1?cbl=5444811&pq-origsite=gscholar)

---

## 💡 Conclusion

By comparing multiple models and applying XAI techniques, we made their decision processes more **transparent** and **interpretable** — which is especially important in sensitive domains like healthcare. With SHAP and LIME, we took a step forward in **trustworthy AI** for medical applications.
