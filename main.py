from loader import Loader
from network import Network

if __name__ == "__main__":
    data = Loader("rsc/heart.csv")
    print(f"dados carregados: features={data.feature_count} linhas={data.row_count}")
    net = Network([data.feature_count, 1])
    print(f"rede criada: {net.summary()}")
