import os
from PIL import Image
import pickle
from img2vec_pytorch import Img2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

img2vec = Img2Vec()

data_dir = 'C:/Users/91962/PycharmProjects/Computer Vision/Weather classification simple model/dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

data = {}

for j, dir_ in enumerate([train_dir, val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_, category)):
            image_path = os.path.join(dir_, category, img_path)
            image = Image.open(image_path).convert('RGB')

            img_features = img2vec.get_vec(image)

            features.append(img_features)
            labels.append(category)

    data[['training_data', 'validation_data'][j]] = features
    data[['training_labels', 'validation_labels'][j]] = labels

#print(data[0:5])

model = RandomForestClassifier(random_state=0)
model.fit(data['training_data'], data['training_labels'])

y_pred = model.predict(data['validation_data'])
score = accuracy_score(y_pred, data['validation_labels'])

print(score)

with open('./model.p', 'wb') as f:
    pickle.dump(model, f)
    f.close()
