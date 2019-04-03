# ROC curve stuff

def get_pos_rate(prob_list, threshold):
    yesses = len( [entry for entry in prob_list if entry > threshold] )
    total = len( prob_list)
    return(yesses/total)

pos_list = list( all_data[all_data['Correct'] == 1]['forest_prob'] )
neg_list = list( all_data[all_data['Correct'] == 0]['forest_prob'] )

print(len(pos_list), len(neg_list))

threshold = []
pos_rate = []
fal_rate = []


for i in np.arange(0.0, 1.0, 0.01):
    threshold = threshold + [i]
    pos_rate = pos_rate + [ get_pos_rate(pos_list, i) ]
    fal_rate = fal_rate + [ get_pos_rate(neg_list, i) ]
    
    from sklearn.metrics import roc_curve

fpos, tpos, thresholds = roc_curve(all_data['Correct'], all_data['forest_prob'])

roc_dat = pd.DataFrame({"Threshold":threshold, "True Positives Rate":pos_rate, "False Positives Rate":fal_rate})
roc_dat

fig = plt.figure()
ax1 = fig.add_subplot(111)

# Plot the data (true positives vs false positives)
ax1.plot(roc_dat['False Positives Rate'], roc_dat['True Positives Rate'])

# Plot the reference line
ax1.plot(roc_dat['False Positives Rate'], roc_dat['False Positives Rate'], color='red', linestyle='--')

# Label stuff
ax1.set_title("ROC Curve - Website classifier")
ax1.set_xlabel("False positive rate")
ax1.set_ylabel("True positive rate")

# Label threshold values on line
thresholds_marked = np.arange(0, 1, 0.1)

# Plots and labels each threshold point
for each in thresholds_marked:
    xpos = get_pos_rate(neg_list, each)
    ypos = get_pos_rate(pos_list, each)
    ax1.plot(xpos, ypos, marker='o', markersize=1, color='black')
    ax1.text(xpos, ypos, str(each))

# Force axes to plot 0-1 ONLY (prevents distortion of representation of area)
ax1.set_xbound(0, 1)
ax1.set_ybound(0, 1)

plt.show()
