import shap
import matplotlib.pyplot as plt

def explain(model, X_sample):
    explainer = shap.DeepExplainer(model, X_sample[:100])
    shap_values = explainer.shap_values(X_sample[:100])
    shap.summary_plot(shap_values, X_sample[:100])
