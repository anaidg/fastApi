# fastApi
taller-fastapi
para ejecutar clonar el repositorio y ejecutar desde la terminal:
uvicorn main:app --reload
y en el browser del navegador http://127.0.0.1:8000
para probar los metodos http://127.0.0.1:8000/docs



para probar el metodo post/predict  ingresar un vector ejemplo 
#
{
  "data_model": [1,1,1,1]
}

para probar el metodo post/predict/ME modelo el cual es un problema de clasificación que indica si una oración tiene las palabras Nueva York o London, usar una lista de palabras por ejemplo

#
{
  "lista": ["new york is great and so is london","i like london better than new york" ]
}
