import os, joblib, shap
import pandas as pd
import matplotlib.pyplot as plt

# load model
pipe = joblib.load("model.pkl")

# data
df = pd.read_csv("data.csv")
X = df.iloc[:, :-1]

# FS transform
X_fs = pipe.named_steps["fs"].transform(X)
rf = pipe.named_steps["model"]

X_fs = pd.DataFrame(
    X_fs,
    columns=X.columns[pipe.named_steps["fs"].get_support()]
)

# SHAP
explainer = shap.TreeExplainer(rf)
sv = explainer(X_fs)

# plot
os.makedirs("shap_results", exist_ok=True)

shap.summary_plot(sv, X_fs, show=False)
plt.savefig("summary.png", dpi=300)
plt.close()

shap.summary_plot(sv, X_fs, plot_type="bar", show=False)
plt.savefig("bar.png", dpi=300)
plt.close()
