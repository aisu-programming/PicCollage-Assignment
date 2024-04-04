import pandas as pd
import matplotlib.pyplot as plt



csv = pd.read_csv("E:/Temporary/新增資料夾/data/responses.csv")
truths = [ vls[1] for vls in csv.values ]

plt.hist(truths, bins=30, rwidth=0.9)
plt.title("Correlation Distribution")
plt.tight_layout()
plt.savefig("Correlation Distribution.png")