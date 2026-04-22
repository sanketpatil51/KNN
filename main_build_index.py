import pandas as pd
from Src.index_builder import build_indexes_with_embeddings


def main():

    print("Loading historical data...")

    historical_df = pd.read_excel("Historic_data.xlsx")

    print("Data loaded successfully")

    print("Building FAISS indexes...")

    build_indexes_with_embeddings(historical_df)

    print("Indexes created successfully.")


if __name__ == "__main__":
    main()