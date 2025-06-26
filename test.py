import pandas as pd
import numpy as np
data = { 'Name' : ['vinay','Ram', 'Rasika', 'Rajesh', 'Rajeshwari'],
         'Age' : [26, 21, 16, 23, 24 ,np.nan]
         }
df = pd.DataFrame(data)
result =  df[df['Age'] > 22]
print(df , "\n")
print(result)
meannn = df['Age'].mean()
summm = df['Age'].sum()
sorttt = df.sort_values(by ='Age')
res=df.[df['Age']==isnull(replace(mean))
print(res)
print(summm)
print(meannn)
print(sorttt)







