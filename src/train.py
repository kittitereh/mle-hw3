import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pymongo import MongoClient
from pyspark.ml.clustering import KMeans


class Train:
    def load_and_process_data(self, data_path: str):
        data = pd.read_csv(data_path, low_memory=False, sep = "\t")
        df = data.dropna(thresh=len(data)*0.6, axis=1)
        #  удалим неинформативные столбцы 
        df1 = df.drop(df.iloc[:, 0:11], axis=1)
        df2 = df1.drop(df1.iloc[:, 1:11], axis=1)
        df2.to_csv('./openfood.csv', index = False)
        clean_csv = spark.read.csv('./openfood.csv',header=True, inferSchema=True)

        return clean_csv

    def scale_data_in_spark(self, clean_csv):
        features = ['fat_100g', 
                    'carbohydrates_100g', 
                    'sugars_100g', 
                    'proteins_100g', 
                    'salt_100g', 
                    'energy_100g']
        assemble = VectorAssembler(inputCols=features, outputCol='features')
        assembled_data = assemble.setHandleInvalid("skip").transform(clean_csv)
        scaler = StandardScaler(inputCol='features', outputCol='scaled')
        scaler_model = scaler.fit(assembled_data)
        scaled_data = scaler_model.transform(assembled_data)

        return scaled_data
    
    def train(self, scaled_data, k = 11):
        kmeans = KMeans(featuresCol='scaled',k=k)
        model = kmeans.fit(scaled_data)
        predictions = model.transform(scaled_data)
        cols = ["pnns_groups_1", "prediction"]
        prediction = predictions.select(*cols)

        return prediction
    
    def save_to_csv(self, prediction):
        prediction_df = prediction.toPandas()
        prediction_df.to_csv("data/openfood_.csv", index=False)
      


if __name__ == "__main__":
    conf = SparkConf().setMaster("local[*]").setAppName("Clastering")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    train = Train()
    clean_csv = train.load_and_process_data("data/data_.csv")
    scaled_data = train.scale_data_in_spark(clean_csv)
    prediction = train.train(scaled_data)
    train.save_to_csv(prediction)
    prediction_df = pd.read_csv("data/openfood_.csv")

    # загрузка в базу данных
    client = MongoClient('mongodb://0.0.0.0:27017/')
    db = client["my_db"]
    db.my_data_new.drop()
    
    for _ ,row in prediction_df.iterrows():
        db.my_data_new.insert_one(row.to_dict())




    



