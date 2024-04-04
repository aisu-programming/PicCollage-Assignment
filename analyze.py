import dotenv
dotenv.load_dotenv(".env")
import os
import pandas as pd
import matplotlib.pyplot as plt



csv = pd.read_csv(f"{os.environ['DATA_DIR']}/responses.csv")
truths = [ vls[1] for vls in csv.values ]

plt.hist(truths, bins=30, rwidth=0.9)
plt.title("Correlation Distribution")
plt.tight_layout()
plt.savefig("Correlation Distribution.png")