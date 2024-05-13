import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython import embed
import numpy as np
from scipy.stats import ks_2samp

# Read the CSV file
df = pd.read_csv('/Users/xie/WORKSPACE/obs/LBT-plot/csv/clean_SDSSDR16Q.csv') # this is the cleaned sample
rl_sdss_df=pd.read_csv('/Users/xie/WORKSPACE/obs/LBT-plot/csv/RACSxVLASSxNVSSxLoTSSxSDSS16q.csv')
#radio_loud_df = pd.read_csv('/Users/xie/WORKSPACE/obs/LBT-plot/csv/RACSxVLASSxNVSSxLoTSSxSDSS16q.csv')
high_z_df = pd.read_csv('/Users/xie/WORKSPACE/obs/LBT-plot/csv/all_high_z_qso.csv')


# Generate sample arrays a and b
sb_lbol=[47.04, 46.99, 47.28, 47.02]
sb_mbh=[9.19, 9.02, 9.14, 8.58]
rl_lbol=[47.57, 47.74, 46.90, 47.22, 47.86, 47.04, 47.54, 46.80, 46.98]
# rl_lbol=[47.06, 47.22, 47.74, 47.39, 47.86, 46.90, 47.04, 47.57, 47.54,46.59,46.80,46.98]
# rl_mbh=[8.70, 9.44, 9.12, 8.87, 9.18, 8.94, 8.75, 9.45,9.19, 9.02, 9.14, 8.58]
rl_mbh=[9.45, 9.12, 8.94, 9.44, 9.18, 8.75, 9.13, 8.43, 8.82]
edd_ratio=[1.06, 3.31, 0.73, 0.49, 3.81, 1.54, 2.05, 1.85, 1.16]

# Extract columns 'a' and 'b' as numpy arrays
sdss_lbols= rl_sdss_df['LOGLBOL'].values
sdss_mbhs = rl_sdss_df['LOGMBH_MGII'].values

# radio_loud_sdss_lbols= radio_loud_df['LOGLBOL'].values
# radio_loud_sdss_mbhs = radio_loud_df['LOGMBH_MGII'].values

# Filter the DataFrame to only include rows with numeric values
high_z_df['BHmass_MgII_log10'] = pd.to_numeric(high_z_df['BHmass_MgII_log10'], errors='coerce')
high_z_df['Lbol_log10'] = pd.to_numeric(high_z_df['Lbol_log10'], errors='coerce')
high_z_df = high_z_df[(high_z_df['BHmass_MgII_log10'].notnull()) & (high_z_df['Lbol_log10'].notnull())]
high_z_mbhs=high_z_df['BHmass_MgII_log10'].values
high_z_lbols=high_z_df['Lbol_log10'].values

high_z_mbhs=np.concatenate([high_z_mbhs, sb_mbh], axis=0)
high_z_lbols=np.concatenate([high_z_lbols, sb_lbol], axis=0)

a=rl_lbol
b = sdss_lbols
_b = b

# Function to find the closest value in b for a given value in a
def find_closest_value(a_val, b):
    return min(b, key=lambda x: abs(x - a_val))

# Create subset b
subset_b = []
subset_b_pos = []
dist=[]
iteration=0
print(np.median(a))
print(np.var(a))
previous_p_value=0
p_value=1
p_values=[]

while True:
    iteration+=1
    previous_p_value=p_value
    for a_val in a:
        closest_val = find_closest_value(a_val, b)
        dist.append(np.abs(closest_val-a_val)**2)
        subset_b.append(closest_val)
        subset_b_pos.append(np.abs(b - closest_val).argmin())
        b = np.delete(b, np.abs(b - closest_val).argmin())  # Remove the closest value from b
    _,p_value=ks_2samp(a,subset_b)
    p_values.append(p_value)
    #if np.sqrt(np.sum(dist)/len(subset_b)) > np.std(a) or (np.median(subset_b)-np.median(a)) > np.std(a) or np.std(subset_b) > np.std(a):
    if p_value<previous_p_value:    
        break

print(f'total distance: {np.sum(dist)}\n total iterations: {iteration}')

def kstest(a,b):
    statistic, p_value = ks_2samp(a, b)
    # print(f"KS statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.1e}")
    print(f"median of our sample:{np.median(a):.2f} median of matched sample:{np.median(b):.2f}")
    print(f"mean of our sample:{np.mean(a):.2f} median of matched sample:{np.mean(b):.2f}")
    print(f"std of our sample:{np.std(a):.2f} median of matched sample:{np.std(b):.2f}")
    

kstest(a, subset_b)

kstest(rl_mbh, sdss_mbhs[subset_b_pos])
lambda_subset_b = 10**((subset_b)-np.log10(((1.26 * 10**38 * 10**(sdss_mbhs[subset_b_pos])))))
kstest(edd_ratio, lambda_subset_b)


# # Plot histograms separately with shared x-axis
# fig, (ax1, ax2) = plt.subplots(2, sharex=True)

# ax1.hist(a, bins=15, alpha=0.5, color='b', label='a')
# ax1.set_ylabel('Frequency')
# ax1.set_title('Histogram of a')

# ax2.hist(subset_b, bins=15, alpha=0.5, color='r', label='Subset of b')
# ax2.set_xlabel('Value')
# ax2.set_ylabel('Frequency')
# ax2.set_title('Histogram of subset of b')

# #embed()

# plt.show()
