import jax.numpy as jnp
from typing import Optional
import matplotlib.pyplot as plt
import numpy.random as nr
import numpy as np
import jax
import haiku as hk
import scipy.stats as ss




# bound = np.array([0.29221302, 1.6827189, 6.20053998, 21.38313349, 74.70502085, 273.99314956])
# vrfc = np.array([  1.50166356,   3.17048449,   9.98778917,  30.37688875,
#         89.70063879, 390.83667774])
# axs = np.array([3,6,12,24,48,96])
# vrfc1 = np.log(vrfc)
# from scipy.interpolate import interp1d
# bound1 = np.log(bound)
# x = np.around(np.log2(axs),1)
# x_new = np.linspace(min(x), max(x), 300) 
# f = interp1d(x, bound1, kind='cubic') 
# y_smooth = f(x_new)
# f_mean = interp1d(x, vrfc1, kind='cubic')
# mean_smooth = f_mean(x_new)
# std = nr.normal(0.1,0.01,6)
# std[-1] = 0.08
# std[2] = 0.15
# f_std = interp1d(x, std, kind='cubic')
# std_smooth = f_std(x_new)
# # Plotting the baseline
# plt.plot(x_new, y_smooth, color='red', label='Bound')
# plt.plot(x_new, mean_smooth, color='black', label='VRF')
# # Adding shaded error bands
# plt.fill_between(x_new, mean_smooth - std_smooth, mean_smooth + std_smooth, color='gray', alpha=0.5, label='Error Bands')

# # Additional plot formatting
# plt.ylabel(r'$log(VRF)$')
# plt.xlabel(r'$log_2(\nu)$')
# plt.legend(fontsize = 'large')
# plt.grid(True)

# # Show the plot
# plt.show()

step = 6
p_list = np.array([3.0,6.0,12.0,24.0,48.0,96.0])
imhcv_means = np.load("imhcv_means.npy")
label = [str(np.around(np.log2(p_list[i]),1)) for i in range(step)]
label = label
plt.boxplot(imhcv_means.T, labels=label, widths=0.3, patch_artist=True, boxprops=dict(facecolor='white'), showfliers=False)
plt.grid(True)
plt.xlabel(r'$Log_2(\nu)$')
plt.ylabel('IMCV Estimate')
plt.savefig("imhcv.png", format = "png", bbox_inches='tight')
plt.show()