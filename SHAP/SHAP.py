import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# load feature
X = pd.read_csv("BorutaFeature.csv")
# load model
rf = joblib.load("RF_model_Boruta.pkl")
#SHAP
explainer = shap.TreeExplainer(rf)
sv = explainer(X)

# plot
shap.summary_plot(sv, X, show=False)
plt.savefig("shap_summary.png", dpi=300)
plt.close()