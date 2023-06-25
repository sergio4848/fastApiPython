from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf

# FastAPI uygulamasını oluşturma
app = FastAPI()


# Gelen veri için model isteği için kullanılacak veri modeli
class PredictionRequest(BaseModel):
    data: list


# Eğitilmiş modeli yükleme
model = tf.keras.models.load_model("trained_model.h5")


# Tahmin isteği işleme
@app.post("/predict")
def predict(request: PredictionRequest):
    # Gelen verileri al
    data = request.data

    # Verileri modelin beklentisine uygun hale getirme
    # Burada gelen veriye özel işlemler yapmanız gerekebilir, örneğin boyut dönüşümü
    processed_data = preprocess_data(data)

    # Verileri modele besleme ve tahmin yapma
    predictions = model.predict(processed_data)

    # Tahmin sonuçlarını döndürme
    return {"predictions": predictions.tolist()}
