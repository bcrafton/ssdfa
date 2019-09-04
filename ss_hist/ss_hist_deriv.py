
import numpy as np
import matplotlib.pyplot as plt

ss = np.load('ss_results.npy', allow_pickle=True).item()
fa = np.load('fa_results.npy', allow_pickle=True).item()

ss_deriv_angle = ss['angle_deriv']
ss_deriv_match = ss['match_deriv']
fa_deriv_angle = fa['angle_deriv']
fa_deriv_match = fa['match_deriv']

idx_deriv = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14]
ss_deriv_angle = ss_deriv_angle[idx_deriv]
fa_deriv_angle = fa_deriv_angle[idx_deriv]
ss_deriv_match = ss_deriv_match[idx_deriv]
fa_deriv_match = fa_deriv_match[idx_deriv]

###############################################

fig, ax = plt.subplots()
bins = len(idx_deriv)
index = np.arange(bins)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, ss_deriv_angle, bar_width, alpha=opacity, color='b', label='SS')
rects2 = plt.bar(index + bar_width, fa_deriv_angle, bar_width, alpha=opacity, color='r', label='FA')

plt.xlabel('Layer')
plt.ylabel('Angle (Degrees)')
plt.title('Angle vs BP')
plt.xticks(index + bar_width, ('9', '8', '7', '6', '5', '4', '3', '2', '1'))
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig('deriv_angle.jpg')

###############################################

fig, ax = plt.subplots()
bins = len(idx_deriv)
index = np.arange(bins)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, ss_deriv_match, bar_width, alpha=opacity, color='b', label='SS')
rects2 = plt.bar(index + bar_width, fa_deriv_match, bar_width, alpha=opacity, color='r', label='FA')

plt.xlabel('Layer')
plt.ylabel('Match (%)')
plt.title('Match with BP')
plt.xticks(index + bar_width, ('9', '8', '7', '6', '5', '4', '3', '2', '1'))
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig('deriv_match.jpg')

###############################################




