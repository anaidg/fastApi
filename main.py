import uvicorn
from fastapi import FastAPI
import pickle
import numpy as np
from typing import List
from pydantic import BaseModel
import algoritms.multiEtiqueta


class ItemList(BaseModel):
    data_model: List[float]

class ItemList2(BaseModel):
    lista:List[str]
    #data_model2: np.array

#class ItemList2(BaseModel):
#    data_model: List[np.array]

# Model
loaded_model = pickle.load(open('./models/Model.pkl', 'rb'))
loaded_model_ME = pickle.load(open('./models/multietiquetas.pkl', 'rb'))

# inicializar app
app = FastAPI()

@app.get('/')
async def index():
    return {"result": "prueba"}

@app.get('/items/{variables}')
async def get_item(variables):
    return {"result": variables}

# ML application
@app.post('/predict/')
async def predict(data:ItemList):
    result = loaded_model.predict(np.array(data.data_model).reshape(1,4))
    return result[0]

@app.post('/predict/ME/')
async def predictME(data:ItemList2):
    result = loaded_model_ME.predict(np.array(data.lista))
    all_labels = algoritms.multiEtiqueta.mlb.inverse_transform(result)
    return all_labels
#prueba


    

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)