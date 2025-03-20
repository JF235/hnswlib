import numpy as np
from tqdm import tqdm
import argparse
import pickle as pkl

if __name__ == "__main__":
    # Total fingers
    
    # Filename using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pklfile", help="Nome do arquivo de entrada")
    args = parser.parse_args()
    file_path = args.pklfile

    try:
        with open(file_path, 'rb') as f:
            name_to_index, score_matrix = pkl.load(f)

        print("shape of score_matrix:", score_matrix.shape)
        print("size of loaded data:", len(name_to_index))
    except FileNotFoundError:
        print("Arquivo não encontrado.")
    except Exception as e:
        print("Erro ao carregar os dados:", e)