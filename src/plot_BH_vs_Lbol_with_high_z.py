import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

# style part
import matplotlib as mpl
from IPython import embed
# Change the default font size for various text elements
mpl.rcParams['font.size'] = 10  # Default font size for text
mpl.rcParams['axes.labelsize'] = 16  # Font size for axis labels
mpl.rcParams['xtick.labelsize'] = 14  # Font size for x-axis tick labels
mpl.rcParams['ytick.labelsize'] = 14  # Font size for y-axis tick labels
mpl.rcParams['legend.fontsize'] = 10  # Font size for legend

# Create the figure and gridspec
fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], wspace=0.05, hspace=0.05)

# Create the axes
ax_scatter = fig.add_subplot(gs[1, 0])

# # Make the ax_scatter square-looking
# ax_scatter.set_aspect('equal', 'box')

ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

# Read the CSV file
df = pd.read_csv('/Users/xie/WORKSPACE/obs/LBT-plot/csv/clean_SDSSDR16Q.csv') # this is the cleaned sample
#radio_loud_df = pd.read_csv('/Users/xie/WORKSPACE/obs/LBT-plot/csv/RACSxVLASSxNVSSxLoTSSxSDSS16q.csv')
high_z_df = pd.read_csv('/Users/xie/WORKSPACE/obs/LBT-plot/csv/all_high_z_qso.csv')

lbt_name=['ILTJ2201+2338*', 'ILTJ2327+2454', 'J0025-0145', 'J0616-1338*', 'J0747+1153', 'J0913+5919', 'J1614+4640', 'J2344+1653']
lbt_lbol=[47.06, 47.22, 47.74, 47.39, 47.86, 46.90, 47.04, 47.57]
lbt_mbh=[8.70, 9.44, 9.12, 8.87, 9.18, 8.94, 8.75, 9.45]
lbt_higher_errors_lbol=[0.21, 0.00, 0.02, 0.13, 0.10, 0.10, 0.14, 0.05]
lbt_lower_errors_lbol=[0.28, 0.00, 0.03, 0.15, 0.11, 0.11, 0.18, 0.06]
lbt_higher_errors_mbh=[0.29, 0.23, 0.05, 0.09, 0.11, 0.09, 0.10, 0.10]
lbt_lower_errors_mbh=[0.33, 0.31, 0.05, 0.08, 0.12, 0.09, 0.12, 0.12]

sb_name=['J1917+5003', 'P067-14', 'P335-15', 'P037-08']
sb_lbol=[47.04, 46.99, 47.28, 47.02]
sb_mbh=[9.19, 9.02, 9.14, 8.58]
sb_higher_errors_mbh=[0.22, 0.08, 0.03, 0.28]
sb_lower_errors_mbh=[0.27, 0.08, 0.03, 0.39]
sb_higher_errors_lbol=[0.07, 0.07, 0.02, 0.05]
sb_lower_errors_lbol=[0.07, 0.07, 0.02, 0.05]

my_sources_name=['P215-16', 'P352-15*', 'J2318-3114', 'J0410-0139']
my_sources_lbol=[47.54,46.59,46.80,46.98]
# my_sources_mbhs=[9.42,9.05,8.95,9.44] # CIV
my_sources_mbhs=[9.13, 8.72, 8.43, 8.82] # MgII
my_sources_rL=[2.7, 1100, 121, 74] 
my_sources_higher_errors_mbh=[0.04, 0.11, 0.13, 0.07] # MgII
my_sources_lower_errors_mbh=[0.04, 0.12, 0.15, 0.07] # MgII
my_sources_higher_errors_lbol=[0.01, 0.01, 0.02, 0.05]
my_sources_lower_errors_lbol=[0.01, 0.01, 0.02, 0.05]

# Extract columns 'a' and 'b' as numpy arrays
sdss_lbols= df['LOGLBOL'].values
sdss_mbhs = df['LOGMBH_MGII'].values

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
_all_radio_loud_mbh=np.concatenate([lbt_mbh,my_sources_mbhs], axis=0)
_all_radio_loud_lbol=np.concatenate([lbt_lbol,my_sources_lbol], axis=0)

df = pd.DataFrame({'BH_mass': sdss_mbhs, 'Lbol': sdss_lbols})
#radio_loud_df = pd.DataFrame({'BH_mass': radio_loud_sdss_mbhs, 'Lbol': radio_loud_sdss_lbols})
high_z_df=pd.DataFrame({'BH_mass': high_z_mbhs, 'Lbol': high_z_lbols})

# Plot the histograms
_, bins, _ = ax_histx.hist(high_z_mbhs, bins=20, edgecolor='black', color='#1E88E5', alpha=0.5)
ax_histx.hist(_all_radio_loud_mbh, bins=bins, edgecolor='black', color='#D81B60', alpha=0.5)
_, bins, _ =ax_histy.hist(high_z_lbols, bins=20, orientation='horizontal', edgecolor='black', color='#1E88E5', alpha=0.5)
ax_histy.hist(_all_radio_loud_lbol, bins=bins, orientation='horizontal', edgecolor='black', color='#D81B60', alpha=0.5)

ax_histx.xaxis.set_tick_params(labelbottom=False)
ax_histx.set_ylabel('Number')
ax_histy.yaxis.set_tick_params(labelleft=False)
# ax_histx.xaxis.set_tick_params(labelleft=False)
# ax_histy.xaxis.set_tick_params(labelleft=False)
# ax_histx.set_xticks([])
# ax_histy.set_xticks([])
# ax_histx.set_yticks([])
# ax_histy.set_yticks([])


# # Draw vertical dashed lines on the histograms
# for i in range(len(my_sources_lbol)):
#     ax_histx.axvline(x=my_sources_mbhs[i], color='#D81B60', linestyle='--')
#     ax_histy.axhline(y=my_sources_lbol[i], color='#D81B60', linestyle='--')
    
# for i in range(len(lbt_lbol)):
#     ax_histx.axvline(x=lbt_mbh[i], color='#D81B60', linestyle='--')
#     ax_histy.axhline(y=lbt_lbol[i], color='#D81B60', linestyle='--')



# plot the scatters
sns.kdeplot(data=df, x='BH_mass', y='Lbol', levels=5, label=r'$\mathrm{DR16Q}$', ax=ax_scatter)
#sns.kdeplot(data=radio_loud_df, x='BH_mass', y='Lbol', levels=5, ax=ax, label=r'$\mathrm{DR16Q}\,\mathrm{radio-loud}$')

# ax_scatter.set_xlim(6.9,10.0)
ax_scatter.set_xlim(6.9,10.2)
# ax_scatter.set_ylim(43.7,47.7)
ax_scatter.set_ylim(43.7,48.4)


# plot Eddington limit
x=np.linspace(0,20,1000)
y=np.log10(1.26*10**38)+x

ax_scatter.plot(x,y, linestyle='--', c='grey')
ax_scatter.text(7.89, 45.61, r'$1{\times}L_{\mathrm{Edd}}$', rotation=np.arctan(3.1/4)*(360/6.28), verticalalignment='bottom', horizontalalignment='right', fontsize=20)

# plot Eddington limit x 0.1
x=np.linspace(0,20,1000)
y=np.log10(0.1*1.26*10**38)+x

ax_scatter.plot(x,y, linestyle='--', c='grey')
ax_scatter.text(7.74, 44.34, r'$0.1{\times}L_{\mathrm{Edd}}$', rotation=np.arctan(3.1/4)*(360/6.28), verticalalignment='bottom', horizontalalignment='right', fontsize=20)

# plot Eddington limit x 0.01
x=np.linspace(0,20,1000)
y=np.log10(0.01*1.26*10**38)+x

ax_scatter.plot(x,y, linestyle='--', c='grey')
ax_scatter.text(8.35, 43.90, r'$0.01{\times}L_{\mathrm{Edd}}$', rotation=np.arctan(3.1/4)*(360/6.28), verticalalignment='bottom', horizontalalignment='right', fontsize=20)

#Scatters
ax_scatter.scatter(my_sources_mbhs, my_sources_lbol, s=70, marker='d', edgecolor='black', zorder=3, c='#D81B60')
ax_scatter.scatter(lbt_mbh, lbt_lbol, s=70, marker='d', edgecolor='black', label='radio-loud sample', zorder=3, c='#D81B60')
ax_scatter.scatter(high_z_mbhs, high_z_lbols, s=50, marker='o', edgecolors='black', alpha=0.5, label='Lit. Quasars z>5.3', c='#1E88E5')


# # plot the source name
# for i in range(len(my_sources_lbol)):
#     ax_scatter.text(my_sources_mbhs[i]-0.3, my_sources_lbol[i], my_sources_name[i], zorder=3, fontweight='bold', color='white', path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
#     import matplotlib.patheffects as path_effects
ax_scatter.errorbar(my_sources_mbhs, my_sources_lbol, xerr=[my_sources_lower_errors_mbh, my_sources_higher_errors_mbh],  \
    yerr=[my_sources_lower_errors_lbol,my_sources_higher_errors_lbol], ls='none', capsize=2, capthick=1, color='black')

# # plot the source name
# for i in range(len(lbt_lbol)):
#     ax_scatter.text(lbt_mbh[i]-0.3, lbt_lbol[i], lbt_name[i], zorder=3, fontweight='bold', color='white', path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
#     import matplotlib.patheffects as path_effects
ax_scatter.errorbar(lbt_mbh, lbt_lbol, xerr=[lbt_lower_errors_mbh, lbt_higher_errors_mbh],  \
    yerr=[lbt_lower_errors_lbol, lbt_higher_errors_lbol], ls='none', capsize=2, capthick=1, color='black')

ax_scatter.set_xticks([7,8,9,10])   
ax_scatter.set_yticks([44,45,46,47,48])   

ax_scatter.set_xticks(np.arange(7,10,0.2), minor=True)
ax_scatter.set_yticks(np.arange(44,47.7,0.2), minor=True)
ax_scatter.tick_params(axis='x', which='minor', length=5)  
ax_scatter.tick_params(axis='x', which='major', length=8)  
ax_scatter.tick_params(axis='y', which='minor', length=5)
ax_scatter.tick_params(axis='y', which='major', length=8) 

# Set labels and title
ax_scatter.set_xlabel(r'$\mathrm{log}M_{\mathrm{BH}}\,(M_{\odot})$', labelpad=10)
ax_scatter.set_ylabel(r'$\mathrm{log}L_{\mathrm{bol}}\,(\mathrm{erg}\,\mathrm{s}^{-1})$', labelpad=10)

ax_scatter.legend()

plt.savefig('./BH_mass_vs_Lbol_hist_with_LBT.pdf', dpi=1000)
#plt.savefig('./BH_mass_vs_Lbol_with_RL_SDSSQ.pdf', dpi=1000)