from config import join_pth
import pandas as pd

df = pd.read_csv(join_pth('df.csv'), nrows=10)
test = pd.read_csv(join_pth('test.csv'), nrows=10)


