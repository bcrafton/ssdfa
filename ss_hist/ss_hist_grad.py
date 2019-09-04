
import numpy as np
import matplotlib.pyplot as plt

ss = np.load('ss_results.npy', allow_pickle=True).item()
fa = np.load('fa_results.npy', allow_pickle=True).item()

ss_grad_angle = ss['angle_gv']
ss_grad_match = ss['match_gv']
fa_grad_angle = fa['angle_gv']
fa_grad_match = fa['match_gv']

idx_grad = [29, 26, 23, 20, 17, 14, 11, 8, 5, 2]
ss_grad_angle = ss_grad_angle[idx_grad]
fa_grad_angle = fa_grad_angle[idx_grad]
ss_grad_match = ss_grad_match[idx_grad]
fa_grad_match = fa_grad_match[idx_grad]

###############################################

fig, ax = plt.subplots()
bins = len(idx_grad)
index = np.arange(bins)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, ss_grad_angle, bar_width, alpha=opacity, color='b', label='SS')
rects2 = plt.bar(index + bar_width, fa_grad_angle, bar_width, alpha=opacity, color='r', label='FA')

plt.xlabel('Layer')
plt.ylabel('Angle (Degrees)')
plt.title('Angle vs BP')
plt.xticks(index + bar_width, ('9', '8', '7', '6', '5', '4', '3', '2', '1'))
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig('grad_angle.jpg')

###############################################

fig, ax = plt.subplots()
bins = len(idx_grad)
index = np.arange(bins)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, ss_grad_match, bar_width, alpha=opacity, color='b', label='SS')
rects2 = plt.bar(index + bar_width, fa_grad_match, bar_width, alpha=opacity, color='r', label='FA')

plt.xlabel('Layer')
plt.ylabel('Match (%)')
plt.title('Match with BP')
plt.xticks(index + bar_width, ('9', '8', '7', '6', '5', '4', '3', '2', '1'))
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig('grad_match.jpg')

###############################################




