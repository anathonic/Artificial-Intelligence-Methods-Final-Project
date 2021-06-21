import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import os

# konwersja string - integer
def atoi(s):
    n = 0
    for i in s:
        n = n*10 + ord(i) - ord("0")
    return n

# tworzenie folderów
main_dirs = ['test','train']
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
os.makedirs('data', exist_ok=True)
for dir in main_dirs:
    os.makedirs(os.path.join('data',dir), exist_ok=True)
    for emotion in emotions:
        os.makedirs(os.path.join('data',dir,emotion), exist_ok=True)

# do zliczania plików png w kazdej kategorii
angry = 0
angry_test = 0
disgusted = 0
disgusted_test = 0
fearful = 0
fearful_test = 0
happy = 0
happy_test = 0
sad = 0
sad_test = 0
surprised = 0
surprised_test = 0
neutral = 0
neutral_test = 0


# załadowanie pliku csv, obliczenie Training
data = pd.read_csv('./fer2013/fer2013.csv')
#print(data['Usage'].value_counts())
training_count = int(data['Usage'].value_counts()[0:1])
png = np.zeros((48,48),dtype=np.uint8)
bytes = 48*48
print("Saving images...")

# czytanie pliku csv linia po linii
for i in tqdm(range(len(data))):
    txt = data['pixels'][i]
    words = txt.split()
    
    # rozmiar obrazów 48x48
    for j in range(bytes):
        xind = j // 48
        yind = j % 48
        png[xind][yind] = atoi(words[j])

    img = Image.fromarray(png)

    # train - czyli wlatuje Training (28709)
    if i < training_count:
        if data['emotion'][i] == 0:
            img.save('data/train/angry/im'+str(angry)+'.png')
            angry += 1
        elif data['emotion'][i] == 1:
            img.save('data/train/disgusted/im'+str(disgusted)+'.png')
            disgusted += 1
        elif data['emotion'][i] == 2:
            img.save('data/train/fearful/im'+str(fearful)+'.png')
            fearful += 1
        elif data['emotion'][i] == 3:
            img.save('data/train/happy/im'+str(happy)+'.png')
            happy += 1
        elif data['emotion'][i] == 4:
            img.save('data/train/sad/im'+str(sad)+'.png')
            sad += 1
        elif data['emotion'][i] == 5:
            img.save('data/train/surprised/im'+str(surprised)+'.png')
            surprised += 1
        elif data['emotion'][i] == 6:
            img.save('data/train/neutral/im'+str(neutral)+'.png')
            neutral += 1

    # test - czyli wlatuje PublicTest i PrivateTest (3589 x 2)
    else:
        if data['emotion'][i] == 0:
            img.save('data/test/angry/im'+str(angry_test)+'.png')
            angry_test += 1
        elif data['emotion'][i] == 1:
            img.save('data/test/disgusted/im'+str(disgusted_test)+'.png')
            disgusted_test += 1
        elif data['emotion'][i] == 2:
            img.save('data/test/fearful/im'+str(fearful_test)+'.png')
            fearful_test += 1
        elif data['emotion'][i] == 3:
            img.save('data/test/happy/im'+str(happy_test)+'.png')
            happy_test += 1
        elif data['emotion'][i] == 4:
            img.save('data/test/sad/im'+str(sad_test)+'.png')
            sad_test += 1
        elif data['emotion'][i] == 5:
            img.save('data/test/surprised/im'+str(surprised_test)+'.png')
            surprised_test += 1
        elif data['emotion'][i] == 6:
            img.save('data/test/neutral/im'+str(neutral_test)+'.png')
            neutral_test += 1

print("Your Dataset is Ready!")
