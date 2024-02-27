import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from collections import Counter
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix

# Visualise word frequencies using word cloud
def word_occurrence_by_group(df):
    token_counts_by_group = {}
    grouped_df = df.groupby('STRIDE')
    for stride_value, group_df in grouped_df:
        all_tokens = []
        for tokens in group_df['NameDesc']:
            all_tokens.extend(tokens)
        token_count = Counter(all_tokens)
        token_counts_by_group[stride_value] = token_count
    return token_counts_by_group

def plot_wc(df):
    print("Generating word cloud...")
    output_dir = '../data/results/wordclouds'
    token_counts_by_group = word_occurrence_by_group(df)
    for stride_value, token_count in token_counts_by_group.items():
        wordcloud = WordCloud(background_color="black").generate_from_frequencies(token_count)
        plt.imshow(wordcloud)
        plt.ion()
        plt.show()
        plt.axis("off")
        sv = ['S','E','D','I','R','T']
        plt.title(f"Word Cloud for STRIDE '{stride_value}': {sv[stride_value]}")
        plt.savefig(f"{output_dir}/wordcloud_{stride_value}.png")
        # plt.pause(1.5)
        # plt.close()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")
    print("Generated all word clouds...")
    return

def plot_graph(hist, model, X_val_padded, y_val, classes):
    output_dir = '../data/results'

    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    y_pred = np.argmax(model.predict(X_val_padded), axis=1)
    cm = confusion_matrix(y_val, y_pred)
    plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes, square=True)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusionmatrix.png", bbox_inches='tight')
    plt.show()
    plt.pause(2)
    return