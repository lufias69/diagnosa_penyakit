from tokenisasi import  tokenisasi as ts
from mysimilarity import similarity_ as sm
from joblib import load
import json
import sys
import os
import pandas as pd
dir_path = os.path.dirname(os.path.realpath(__file__))
deli = ["."]

with open(dir_path+"/model/delimiter.json","r") as f:
    delimiter_ = json.load(f)
with open(dir_path+"/model/vector_model.json","r") as f:
    vector_model = json.load(f)

with open(dir_path+"/model/penyakit.json","r") as f:
    penyakit = json.load(f)

filename = dir_path +'/model/tfidf_model.sav'
tfidf = load(filename)

filename = dir_path +'/model/knn_model_detail.sav'
knn_detail = load(filename)

filename = dir_path +'/model/knn_model_diagnosa.sav'
knn_diagnosa = load(filename)
# tfidf = load(dir_path+"/model/tfidf_model.sav")
# # print(tfidf)
def tovec(x):
    return tfidf.transform([x]).toarray()[0]

def prediksi(x, n=10, delimiter = deli, metode = "knn"):
    x = x.lower()

    if type(delimiter) is not list:
        delimiter = str(delimiter).split()
    delimiter = list(set(delimiter_+delimiter))
    x = ts.ngramku(x, n=n, delimiter = delimiter)['string']
    # print(x)
    to_vector = tfidf.transform([x]).toarray()[0] #tfidf.transform(new_x).toarray()

    if metode == "cosine":
        list_similarity = list()
        for i in vector_model:
            kemiripan = sm.cosine_similarity(i,to_vector)
            list_similarity.append(kemiripan)
        dict_hasil = {
            "penyakit" : penyakit["nama_penyakit"],
            "detail":penyakit["detail"],
            "Jenis_Infeksi":penyakit['Jenis_Infeksi'],
            "similarity":list_similarity
            
        }
        df = pd.DataFrame.from_dict(dict_hasil)
        sorted_ =  df.sort_values(by=['similarity'], ascending=False)
        diagnosa = sorted_['penyakit'].tolist()[0]
        similarity = sorted_['similarity'].tolist()[0]
        detail = sorted_['detail'].tolist()[0]
        Jenis_Infeksi =  sorted_['Jenis_Infeksi'].tolist()[0]
        return {"jenis infeksi":Jenis_Infeksi,"penyakit":diagnosa, "similarity":similarity, "detail penyakit":detail, "data":sorted_, "gejala":x}

    elif metode == "knn":
        dia = knn_diagnosa.predict([to_vector])[0]
        det = knn_detail.predict([to_vector])[0]
        return {"penyakit":dia, "detail penyakit":det}





