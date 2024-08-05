# References
# https://stackoverflow.com/questions/14270391/how-to-plot-multiple-bars-grouped - how to plot bar graphs
'''Graphs each accuracies'''
import matplotlib.pyplot as plt
import numpy as np
import os

# Dictionary of accuracies
accuracies_titles = ['Accuracies of 115 Classes Models', 'Accuracies of 115 Classes Models - Edited Features', 'Accuracies of 34 Classes Models','Accuracies of 34 Classes Models - Edited Features']
top_k_titles = ['Top-K Accuracies of 115 Classes Models', 'Top-K Accuracies of 115 Classes Models - Edited Features', 'Top-K Accuracies of 34 Classes Models','Top-K Accuracies of 34 Classes Models - Edited Features']
accuracies = []
top_k_acc = []

# dictionaries from project_main.py
# all
accuracies.append({'Comparisons': [0.008695652173913044, 0.03573540280857354], 'Feedforward Neural Network': [0.03634907489147153, 0.037065779748706576], 'Decision Trees': [0.8856378379854203, 0.21016260162601627], 'K-Nearest Neighbors': [0.52902738466859, 0.08337028824833703]})
top_k_acc.append({'Comparisons': [0.05217391304347826, 0.21786770140428677], 'Feedforward Neural Network': [0.21352578745711193, 0.2131929046563193], 'Decision Trees': [0.9999817981597939, 0.2900591278640059], 'K-Nearest Neighbors': [0.9701944866626016, 0.17204360679970437]})

accuracies.append({'Comparisons': [0.008695652173913044, 0.03573540280857354], 'Feedforward Neural Network': [0.030761109948215765, 0.02993348115299335], 'Decision Trees': [0.8658888413618617, 0.13484848484848486], 'K-Nearest Neighbors': [0.5368723777973953, 0.1057280118255728]})
top_k_acc.append({'Comparisons': [0.05217391304347826, 0.21786770140428677], 'Feedforward Neural Network': [0.21345298009628774, 0.21334072431633408], 'Decision Trees': [0.9999635963195879, 0.2302660753880266], 'K-Nearest Neighbors': [0.9627226312580202, 0.20306725794530672]})

# short
accuracies.append({'Comparisons': [0.029411764705882353, 0.1278640059127864], 'Feedforward Neural Network': [0.12814095505055562, 0.1278640059127864], 'Decision Trees': [0.9262097398046942, 0.34079822616407984], 'K-Nearest Neighbors': [0.588310778219678, 0.16193643754619363]})
top_k_acc.append({'Comparisons': [0.058823529411764705, 0.25068366592756836], 'Feedforward Neural Network': [0.21399903530246908, 0.2122320768662232], 'Decision Trees': [0.9900708961676026, 0.3737620103473762], 'K-Nearest Neighbors': [0.9820529855568398, 0.28590169992609016]})

accuracies.append({'Comparisons': [0.029411764705882353, 0.1278640059127864], 'Feedforward Neural Network': [0.12152458613565832, 0.12281966001478196], 'Decision Trees': [0.9117756805213008, 0.2668329637841833], 'K-Nearest Neighbors': [0.5992318823433049, 0.19336659275683665]})
top_k_acc.append({'Comparisons': [0.058823529411764705, 0.25068366592756836], 'Feedforward Neural Network': [0.24966554118621392, 0.25068366592756836], 'Decision Trees': [0.9880868955851436, 0.3086659275683666], 'K-Nearest Neighbors': [0.9782032963532613, 0.32718033998521806]})


def plot_bar_graphs(data, title, ylabel, is_top_k=False, filepath=None):
    models = list(data.keys())
    values = np.array(list(data.values())) * 100

    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.35
    index = np.arange(len(models))
    comparison_colors = ['#FF6B6B', '#F9D5E5'] if not is_top_k else ['#A3DE83', '#61C0BF']
    eval_colors = ['#FFA07A', '#FFD700'] if not is_top_k else ['#FFB6C1', '#AFEEEE']

    for i in range(len(models)):
        if i == 0:  # Comparisons 
            label1 = 'Random Chance' if not is_top_k else 'Random Chance (Top K)'
            label2 = 'Majority Classifier' if not is_top_k else 'Top K Classifier'
            ax.bar(index[i], values[i][0], bar_width / 2, label=label1, color=comparison_colors[0])
            ax.bar(index[i] + bar_width / 2, values[i][1], bar_width / 2, label=label2, color=comparison_colors[1])
            ax.text(index[i], values[i][0] + 1, f"{values[i][0]:.2f}%", ha='center', va='bottom')
            ax.text(index[i] + bar_width / 2, values[i][1] + 1, f"{values[i][1]:.2f}%", ha='center', va='bottom')
        else:  # Train and testing bars
            ax.bar(index[i], values[i][0], bar_width / 2, label='Training Set' if i == 1 else None, color=eval_colors[0])
            ax.bar(index[i] + bar_width / 2, values[i][1], bar_width / 2, label='Test Set' if i == 1 else None, color=eval_colors[1])
            ax.text(index[i], values[i][0] + 1, f"{values[i][0]:.2f}%", ha='center', va='bottom')
            ax.text(index[i] + bar_width / 2, values[i][1] + 1, f"{values[i][1]:.2f}%", ha='center', va='bottom')

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
        plt.close()
    else:
        plt.show()


for acc, title in zip(accuracies, accuracies_titles):
    filename = f"{title.replace(' ', '_').lower()}.png"
    filepath = os.path.join(r"C:\Users\gwend\UofA\CMPUT 466\Final Mini Project 466\Graphs", filename)
    plot_bar_graphs(acc, title, 'Accuracy', filepath=filepath)

for acc, title in zip(top_k_acc, top_k_titles):
    filename = f"{title.replace(' ', '_').lower()}.png"
    filepath = os.path.join(r"C:\Users\gwend\UofA\CMPUT 466\Final Mini Project 466\Graphs", filename)
    plot_bar_graphs(acc, title, 'Top K Accuracy', is_top_k=True, filepath=filepath)
