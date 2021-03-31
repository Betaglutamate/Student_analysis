import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('data/OD_measures2.csv')

## subtract minimum value
subtractor = data.iloc[0:10][data.columns[1:61]].min()
for num, column in enumerate(data.columns[1:61]):
    data[column] = data[column].apply(lambda x: x-subtractor.values[num]) 
    #loop through each column of the data frame and subtract the value

dfLong = data.melt(id_vars="Time.min", value_vars=data.columns[1:61])

# add groups
dfLong["group"] = dfLong["variable"].apply(lambda x: x[0:7])

# add log od
dfLong['log-OD'] = np.log(dfLong['value'])


# align dataframe

def align_data(df):
    aligned_df = df[df['value'] > 0.01].copy()
    aligned_df['new_time'] = df.iloc[0:len(aligned_df)]['Time.min'].values
    return aligned_df

aligned_df = []
for variable, df in dfLong.groupby(dfLong['variable']):
    temp_df = align_data(df)
    aligned_df.append(temp_df)

aligned_df = pd.concat(aligned_df).reset_index(drop=True)



# calculate linear regression
def gen_slope(df):
    
    df_new = df.reset_index(drop= True)
    windows = []
    for row in range(len(df_new.index) -7):
        window = df_new.iloc[row : row+8]
        windows.append(window)
        
    output1 = []
    for i, df in enumerate(windows):
        x = df['Time.min']
        y = df['log-OD']
        res = stats.linregress(x, y)
        slope,r= res.slope*60,res.rvalue
        output1.append([slope,r])
        
    slope_r = pd.DataFrame(output1,columns=['growthrate','R-value'])
    
    final_df = df_new.join(slope_r)
    return final_df


calculated_gr = []
for variable, df in aligned_df.groupby(aligned_df['variable']):
    temp_df = gen_slope(df)
    calculated_gr.append(temp_df)

final_df = pd.concat(calculated_gr).reset_index(drop=True).dropna()

plot_variables = [[name, df] for name, df in final_df.groupby(final_df['variable'])]

def plot_gr_vs_time (name, df):
    from pathlib import Path
    Path("data/plots").mkdir(parents=True, exist_ok=True)
    path = "data/plots/"
    fig, ax = plt.subplots()
    current_variable = name

    sns.scatterplot(data=df, x='new_time', y='growthrate', color = 'g')
    ax2 = plt.twinx()
    sns.scatterplot(data=df, x='new_time', y='log-OD', ax=ax2)
    ax.set_xlim(0, 500)
    ax.set_title(name)
    plt.savefig(f"data/plots/GR_plot{current_variable}.png")
    plt.close()

[plot_gr_vs_time(name, df) for name, df in plot_variables]
