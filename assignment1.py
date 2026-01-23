import numpy as np
import pandas as pd


data = pd.read_table("assign1_data.txt", sep = "\s+")
#Resolved spaces in dataframe by using https://www.geeksforgeeks.org/pandas/how-to-read-space-delimited-files-in-pandas/

df = data.copy()

#Checked to see if Pandas has a mean function: https://www.geeksforgeeks.org/pandas/how-to-read-space-delimited-files-in-pandas/
y_avg = df["y"].mean()
x1_avg = data["x1"].mean()
x2_avg = data["x2"].mean()

#Note: these are Series
x1_sub = df["x1"] - x1_avg
x2_sub = df["x2"] - x2_avg
y_sub = df["y"]- y_avg

m_x1 = (sum((x1_sub * y_sub)))/((sum(x1_sub**2)))
m_x2 = (sum((x2_sub * y_sub)))/((sum(x2_sub**2)))

b_x1 = y_avg - m_x1*x1_avg
b_x2 = y_avg - m_x2*x2_avg

lstsq_x1 = sum((df["y"]-(m_x1*df["x1"]+b_x1))**2)
lstsq_x2 = sum((df["y"]-(m_x2*df["x2"]+b_x2))**2)

print("Least Squares for x1: " + str(lstsq_x1))
print("Least Squares for x2: " + str(lstsq_x2))

#Part 2
A = np.vstack([df["x1"], df["x2"], np.ones(len(df["x1"]))]).T
w1, w2, b = np.linalg.lstsq(A, df["y"])[0]
print("w1: " + str(w1))
print("w2: " + str(w2))
print("b: " + str(b))

#Part 3, used https://www.geeksforgeeks.org/python/ways-to-apply-an-if-condition-in-pandas-dataframe/ to find a simple way to incorp. coditionals into filtering a df.
df_copy = df.copy()
df_copy["reg"] = w1*df["x1"]+w2*df["x2"]+b
df_copy["z_calc"] = np.where(df_copy["reg"] > 0, 1, 0)


pct = (len((df_copy["z_calc"] > 0))/len((df["z"]>0))) * 100

print("Percentage Match: " + str(pct) + "%")


