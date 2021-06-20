import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import os

# konwertowanie stringa do integera
def atoi(s):
    n = 0
    for i in s:
        n = n*10 + ord(i) - ord("0")
    return n

# tworzenie folderów
outer_name =  'emotions'
inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
os.makedirs('data', exist_ok=True)
os.makedirs(os.path.join('data',outer_name), exist_ok=True)
for inner_name in inner_names:
    os.makedirs(os.path.join('data',outer_name,inner_name), exist_ok=True)

# do zliczania zdjęć kazdej kategorii od 0
angry = 0
disgusted = 0
fearful = 0
happy = 0
sad = 0
surprised = 0
neutral = 0
angry_test = 0
disgusted_test = 0
fearful_test = 0
happy_test = 0
sad_test = 0
surprised_test = 0
neutral_test = 0

#zliczanie ilość plików z 'Training' Usage do pętli
data = pd.read_csv('./fer2013/fer2013.csv')
x = int(data['Usage'].value_counts()[0:1])
mat = np.zeros((48,48),dtype=np.uint8)
print("Saving images...")
bytes = 48*48
# czytanie pliku csv z datasetem
for i in tqdm(range(len(data))):
    txt = data['pixels'][i]
    words = txt.split()
    
# ustawienie rozmiaru zdjęć na 48x48
    for j in range(bytes):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])

    img = Image.fromarray(mat)

# rozpakowywanie datasetu do podfolderów
    if i < x:
        if data['emotion'][i] == 0:
            img.save('data/emotions/angry/im'+str(angry)+'.png')
            angry += 1
        elif data['emotion'][i] == 1:
            img.save('data/emotions/disgusted/im'+str(disgusted)+'.png')
            disgusted += 1
        elif data['emotion'][i] == 2:
            img.save('data/emotions/fearful/im'+str(fearful)+'.png')
            fearful += 1
        elif data['emotion'][i] == 3:
            img.save('data/emotions/happy/im'+str(happy)+'.png')
            happy += 1
        elif data['emotion'][i] == 4:
            img.save('data/emotions/sad/im'+str(sad)+'.png')
            sad += 1
        elif data['emotion'][i] == 5:
            img.save('data/emotions/surprised/im'+str(surprised)+'.png')
            surprised += 1
        elif data['emotion'][i] == 6:
            img.save('data/emotions/neutral/im'+str(neutral)+'.png')
            neutral += 1

print("Your Dataset is Ready!")
