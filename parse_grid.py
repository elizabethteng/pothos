#!/software/et_env/bin python


import numpy as np
import os
import pdspy.modeling as modeling

dictionary=np.load("./etgrid/et_dictionary.npy",allow_pickle=True)    
directory="./etgrid/models/"

failed=[]
seds_dict=[]
seds=[]

for i in range(len(dictionary)):
    filename=dictionary[i]['filename']
    if filename in os.listdir(directory):
        entry=dictionary[i]
        model=modeling.YSOModel()
        model.read_yso(directory+filename)
        entry['seds'] = np.array(np.log10(model.spectra["SED"].flux))  
        print("adding "+str(i)+" to dictionary")
        seds_dict.append(entry)
        seds.append(np.array(np.log10(model.spectra["SED"].flux)))
    else:
        print(str(i)+" failed")
        failed.append(i)
                         

print(str(len(seds_dict))+" added to seds_dict, "+str(len(failed))+ " failed")

np.save("./etgrid/et_dictionary_seds.npy",seds_dict)
np.save("./failed.npy",failed)
np.save("./etgrid/seds.npy",seds)


