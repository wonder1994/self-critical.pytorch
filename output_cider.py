import pickle
with open('reinforce/5em5_10_demean/histories_fc.pkl') as a:
    data_rf_demean = pickle.load(a)
cider_rf_demean={}

for i in range(20):
    print(337000 + i*1000)
    print(data_rf_demean['val_result_history'][337000 + i *1000]['lang_stats']['CIDEr'])
    if data_rf_demean['val_result_history'][337000 + i *1000]['lang_stats']['CIDEr'] is None:
        break
    else:
        cider_rf_demean[337000 + i *1000] = data_rf_demean['val_result_history'][337000 + i *1000]['lang_stats']['CIDEr']


with open('reinforce/5em5_10_nodemean/histories_fc.pkl') as a:
with open('histories_fc.pkl') as a:
    data_rf_nodemean = pickle.load(a)
cider_rf_nodemean={}

for i in range(200):
    print(337000 + i*1000)
    print(data_rf_nodemean['val_result_history'][337000 + i *1000]['lang_stats'])
    if data_rf_nodemean['val_result_history'][337000 + i *1000]['lang_stats'] is None:
        break
    else:
        cider_rf_nodemean[337000 + i *1000] = data_rf_nodemean['val_result_history'][337000 + i *1000]['lang_stats']
print(data_rf_nodemean['val_result_history'][338000]['lang_stats'])