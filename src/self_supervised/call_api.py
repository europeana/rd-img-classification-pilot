import os
import requests

class ModelAPI():

    def __init__(self,endpoint_name,port = 5000):
        self.url = f"http://rnd-3.eanadev.org:{port}/{endpoint_name}"

    def is_active(self):
        try:
            requests.get(self.url).json() 
            active = True
        except:
            active = False

        return active

    def get_data(self):

        resp = requests.get(self.url).json()
        return resp

        
    def predict(self,img_path_list):
        if not self.is_active():
            return None

        if isinstance(img_path_list,str):
            img_path_list = [img_path_list]

        pred_dict = {}
        bs = 10

        path = img_path_list[0]

        file_dict = {f'file':open(path,'rb')}
        resp = requests.post(self.url,files=file_dict).json() 
        #print(resp)
        pred_dict.update(resp)

        return pred_dict

model = ModelAPI('img_recommendation',port = 5000)

data = model.get_data()

image_list = data['fpaths']


img_path = '/home/jcejudo/projects/image_classification/data/single_label/images_training/building'


img_path_list = [os.path.join(img_path,fname) for fname in os.listdir(img_path)]


#print(img_path_list)
#pred_dict = model.predict(img_path_list,XAI=False)
pred_dict = model.predict(image_list[0])

print(pred_dict)


