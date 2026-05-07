import numpy as np
from sklearn.linear_model import LogisticRegression
from deap_pipeline import DEAPPipeline
from train_transformer import DEAPDataset

DATA_DIR = r'c:\Users\sambh\Downloads\archive (9)\deap-dataset\data_preprocessed_python'
TRAIN_SUBJECTS = [f's{i:02d}' for i in range(1, 25)]
TEST_SUBJECTS = [f's{i:02d}' for i in range(25, 33)]

pipeline = DEAPPipeline(data_dir=DATA_DIR, window_size_sec=30.0, overlap_sec=15.0, remove_baseline=True, extract_eeg_features=True, extract_hrv=True)
train_ds = DEAPDataset(DATA_DIR, TRAIN_SUBJECTS, pipeline)
test_ds = DEAPDataset(DATA_DIR, TEST_SUBJECTS, pipeline)

X_train = np.concatenate([train_ds.X_eeg, train_ds.X_ecg], axis=1)
X_test = np.concatenate([test_ds.X_eeg, test_ds.X_ecg], axis=1)
y_train = train_ds.y
y_test = test_ds.y

clf = LogisticRegression(max_iter=2000, multi_class='multinomial', solver='saga')
clf.fit(X_train, y_train)
print('train acc', clf.score(X_train, y_train))
print('test acc', clf.score(X_test, y_test))
print('test dist', np.bincount(clf.predict(X_test), minlength=5).tolist())
