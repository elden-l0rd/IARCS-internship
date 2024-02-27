import pandas as pd
from modules import preprocessing as pre
from modules import extract_keywords as ek
from modules import visualise as vis
import model

PATH = '../data/external/mitre-classified.xlsx'
df = pd.read_excel(PATH)

'''
change labels for simplicity
S  T  R  I  D  E  (mapping)
0  5  4  3  2  1
'''

df_train, df_test, df_dev = pre.split_data(df)

col_toDrop = ['Ref', 'Name', 'Desc', 'Confidentiality', 'Integrity', 'Availability', 'Ease Of Exploitation', 'References', 'Unnamed: 0']
df_train = df_train.reset_index(drop=True).drop(columns=col_toDrop)
df_test = df_test.reset_index(drop=True).drop(columns=col_toDrop)
df_dev = df_dev.reset_index(drop=True).drop(columns=col_toDrop)
print("Data split:")
print(f"df_train:\n{df_train['STRIDE'].value_counts()}\n")
print(f"df_dev:\n{df_dev['STRIDE'].value_counts()}\n")
print(f"df_test:\n{df_test['STRIDE'].value_counts()}")
print("=========================================\n")

# trivially extract keywords
df_train = pre.text_preprocessing(df_train)
df_test = pre.text_preprocessing(df_test)
df_dev = pre.text_preprocessing(df_dev)
print("Trivial text preprocessing:")
print(f"df_train:\n{df_train.head(1)}\n")

# obtain better set of keywords
df_train = ek.better_keywords(df_train)
vis.plot_wc(df_train)

X_train_tfidf, X_test_tfidf, X_val_tfidf, y_train, y_test, y_val = model.vectorize(df_train, df_test, df_dev)

model6 = model.initialise_model(HIDDEN_UNITS=128,
                               NUM_CLASSES=6,
                               VOCAB_SIZE=X_train_tfidf.shape[1])
NUM_EPOCHS = 50
BATCH_SIZE = 16
CLASSES = [0,1,2,3,4,5]
hist6, model6 = model.train_loop(model=model6,
                                 X_train_tfidf=X_train_tfidf,
                                 y_train=y_train,
                                 X_val_tfidf=X_val_tfidf,
                                 y_val=y_val,
                                 NUM_EPOCHS=NUM_EPOCHS,
                                 BATCH_SIZE=BATCH_SIZE)

vis.plot_graph(hist=hist6,
               model=model6,
               X_val_padded=X_test_tfidf,
               y_val=y_test,
               classes=CLASSES)