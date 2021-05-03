import json
import os
import sys
import numpy as np
dic = ["middle","high"]
def race2reco(filename,dirc):
    dataset = []
    middle_dataset = []
    high_dataset = []
    filepath = os.path.join(dirc,filename)
    try:
        data  = json.load(open(filepath))
    except:
        print (filepath)
        exit()
    passage = data["article"]
    passage_id = dirc.split("/")[-1]+"_"+filename.split(".")[0]
    for query_id,query_text in enumerate(data["questions"]):
        options = data["options"][query_id]
        option_text = "|".join(options)
        #answer = options[ord(data["answers"][query_id])-ord("A")]
        answer = ord(data["answers"][query_id])-ord("A")
        _id = "{}_{}".format(passage_id,query_id)
        data_ = {"alternatives":option_text,"passage":passage,"query_id":_id,"answer":answer,"query":query_text}
        dataset.append(data_)
        if passage_id.startswith("middle"):
            middle_dataset.append(data_)
        else:
            high_dataset.append(data_)
    return dataset,middle_dataset,high_dataset
if __name__ == "__main__":
    dataset=[]
    dataset_middle = []
    dataset_high = []
    for d in dic:
        dirc = os.path.join(sys.argv[1],d)
        for filename in os.listdir(dirc):
            d,m,h = race2reco(filename,dirc)
            dataset.extend(d)
            dataset_middle.extend(m)
            dataset_high.extend(h)
    np.random.shuffle(dataset)
    json.dump(dataset,open(sys.argv[1]+".json","w"),indent=4)
    json.dump(dataset_middle,open(sys.argv[1]+"-middle.json","w"),indent=4)
    json.dump(dataset_high,open(sys.argv[1]+"-high.json","w"),indent=4)
