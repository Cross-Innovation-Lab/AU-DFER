import os
from glob import glob
emo_sets = glob(os.path.join('./DFEW_*.txt'))
AU_txt = './AU_DFEW.txt'
AU_dict = {}
with open(AU_txt, "r", encoding="utf-8") as f:
    for line in f:
        k = line[1:6]+'/'+line.split('csv/')[1][:-37]
        AU_dict[k] = line[-36:]
for file in emo_sets:
    if 'AU.txt' in file:
        continue
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            sample_id = line[-11:-6]
            #print(sample_id)
            for i in range(1,17):
                file_data+=sample_id+' '+str(i)+' '+AU_dict[sample_id+'/'+str(i)]
    with open(file.replace('.txt','_AU.txt'), "w", encoding="utf-8") as f:
        f.write(file_data)