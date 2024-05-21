import numpy as np
import fetchmaker
from scipy.stats import binom_test
from scipy.stats import f_oneway    # ANOVA Test
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency

# fetchmaker attributes
fetchmaker.get_weight('rottweiler')
fetchmaker.get_tail_length
fetchmaker.get_age
fetchmaker.get_color
fetchmaker.get_is_rescue


# rettweiler tail length
rottweiler_tl = fetchmaker.get_tail_length('rottweiler')
print(np.mean(rottweiler_tl))
print(np.std(rottweiler_tl))

# rescued whippet analysis
whippet_rescue = fetchmaker.get_is_rescue('whippet')
# print(whippet_rescue)
num_whippet_rescues = np.count_nonzero(whippet_rescue)
print(num_whippet_rescues)

num_whippets = np.size(whippet_rescue)
print(num_whippets)

pval = binom_test((num_whippet_rescues / num_whippets) * num_whippets, num_whippets, 0.08)
print('P-Value:', pval)    # p-value: 0.00058


# weight analysis: mid_sized dog breeds: whippet, terrier, pitbull
w_whippet = fetchmaker.get_weight('whippet')
w_terrier = fetchmaker.get_weight('terrier')
w_pitbull = fetchmaker.get_weight('pitbull')

# perform Anova Test analysis for dog's weights
tstat, pval = f_oneway(w_whippet, w_terrier, w_pitbull)
print(pval)

# perfrom Tukey's Range Test
v = np.concatenate([w_whippet, w_terrier, w_pitbull])
labels = ['w_whippet'] * len(w_whippet) + ['w_terrier'] * len(w_terrier) + ['w_pitbull'] * len(w_pitbull)
tukey_results = pairwise_tukeyhsd(v, labels, 0.05)
print(tukey_results)

# get dog's breed colors
poodle_colors = fetchmaker.get_color('poodle')
shihtzu_colors = fetchmaker.get_color('shihtzu')

# create pivot table 'Poodle' 'Shihtzu' and colors for Chi Square
color_table = [
  [np.count_nonzero(poodle_colors == 'black'), np.count_nonzero(shihtzu_colors == 'black')],
  [np.count_nonzero(poodle_colors == 'brown'), np.count_nonzero(shihtzu_colors == 'brown')],
  [np.count_nonzero(poodle_colors == 'gold'), np.count_nonzero(shihtzu_colors == 'gold')],
  [np.count_nonzero(poodle_colors == 'grey'), np.count_nonzero(shihtzu_colors == 'grey')],
  [np.count_nonzero(poodle_colors == 'white'), np.count_nonzero(shihtzu_colors == 'white')]
  ]

# perform Chi Square analysis
chi2, pval, dof, expected = chi2_contingency(color_table)
print(pval)
