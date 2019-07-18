import pandas as pd
from fuzzywuzzy import fuzz as sim
def prediksi(gejala, namafile):
#     print(namafile)
    data = pd.read_excel(namafile)
    gejala_list = data["Tanda_dan_Gejala"].tolist()
    diagnosa_list = data["Nama Penyakit"].tolist()
    detail = data["Detail Nama Penyakit"].tolist()
#     cek = gejala
#     data_latih = ["a", "b"]
#     diagnosa = ['aa','bb']
    ratio=list()
    for i in gejala_list:
        # rat = sim.token_set_ratio(i, gejala)
        rat = sim.partial_ratio(i, gejala)
        ratio.append(rat)

    dict_ = {
        "similarity":ratio,
        "diagnosa":diagnosa_list,
        "detail":detail
    }


    dataf = pd.DataFrame.from_dict(dict_)
#     print(dataf)
#     sorted_ = dataf.sort_values(by=['similarity'], descending=True)
    sorted_ =  dataf.sort_values(by=['similarity'], ascending=False)
#     print(sorted_)
    diagnosa = sorted_['diagnosa'].tolist()[0]
    similarity = sorted_['similarity'].tolist()[0]
    detail = sorted_['detail'].tolist()[0]
    
    return [diagnosa, similarity, detail, sorted_]
