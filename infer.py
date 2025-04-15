import pickle
from img2vec_pytorch import Img2Vec
from PIL import Image

with open('./model.p', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

image_path = 'C:/Users/91962/PycharmProjects/Computer Vision/Weather classification simple model/test_class.jpeg'

img = Image.open(image_path)

features = img2vec.get_vec(img)

pred = model.predict([features])

print(pred)