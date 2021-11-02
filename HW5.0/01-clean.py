import glob
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    files = glob.glob("data/*")
    features = list()
    target = list()
    encoder = LabelEncoder()
    for label in files:
        file = open(label, 'r')
        content = file.read()
        sentences = sent_tokenize(content)
        for sentence in sentences:
            features.append(sentence)
            target.append(label)
        file.close()
    target = encoder.fit_transform(target)
    np.savez('clean01_processed.npz', features=features, target=target, encoder=encoder)
