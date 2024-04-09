from src.model.knn import train_and_predict_knn

def main():
    #KNN
    #MAPE
    train_and_predict_knn("src/data/dataframe2.csv")

if __name__ == "__main__":
    main()