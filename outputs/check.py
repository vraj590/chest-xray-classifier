# rough estimate — look at your training log
import pandas as pd
df = pd.read_csv("training_log.csv")
print(df[["epoch", "elapsed_s", "val_auc", "val_acc"]])