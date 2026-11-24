import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("submission.csv")

print(df.columns.tolist(), "\n")
print(df["lithology"].unique())
print(df.duplicated(["depth", "lithology"]).sum())

plt.scatter(df["depth"], df["lithology"], alpha=0.2)
plt.xlabel("Lithology")
plt.ylabel("Depth")
plt.show()


