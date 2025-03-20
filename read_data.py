import numpy as np
from tqdm import tqdm
import argparse
import pickle as pkl

def read_similarity_data(filepath):

    # Dicionário para mapear nomes para índices
    name_to_index = {}
    # Lista para armazenar as triplas (index1, index2, score)
    triplets = []
    
    # Primeira passagem: coletar todos os nomes únicos
    with open(filepath, 'r') as file:
        count = 0
        for line in tqdm(file):
            # Dividir a linha em partes
            parts = line.strip().split()
            if len(parts) < 6:  # Verificar se temos pelo menos 6 colunas
                continue  # Ignora linhas que não estão no formato esperado
                
            name1, name2, score_str = parts[0], parts[1], parts[2]
            # Ignoramos os três valores adicionais (parts[3], parts[4], parts[5])
            
            # Adicionar nomes ao dicionário se ainda não existirem
            if name1 not in name_to_index:
                name_to_index[name1] = len(name_to_index)
            if name2 not in name_to_index:
                name_to_index[name2] = len(name_to_index)
            
            # Converter score para float e armazenar a tripla
            try:
                score = float(score_str)
                index1 = name_to_index[name1]
                index2 = name_to_index[name2]
                triplets.append((index1, index2, score))
            except ValueError:
                # Ignora linhas onde o score não é um número válido
                continue
        
            count += 1
            # if count == 1000:
            #     break
    
    # Criar matriz de scores (inicialmente com zeros)
    n = len(name_to_index)
    score_matrix = np.zeros((n, n))
    
    # Preencher a matriz com os scores
    for i1, i2, score in tqdm(triplets):
        score_matrix[i1, i2] = score
        # Se a similaridade for simétrica, descomente a linha abaixo
        score_matrix[i2, i1] = score
    
    return name_to_index, score_matrix.astype(np.float32)

# Exemplo de uso
if __name__ == "__main__":
 
    # Filename using argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Nome do arquivo de entrada")
    args = parser.parse_args()
    file_path = args.filename

    try:
        name_to_index, score_matrix = read_similarity_data(file_path)

        np.save('score_matrix.npy', score_matrix)

        result = (name_to_index, score_matrix)

        # Salvar o resultado em um arquivo pickle
        with open('result.pkl', 'wb') as f:
            pkl.dump(result, f)
        print("Dados lidos e salvos com sucesso.")

        print("Exemplo de mapeamento de nome para índice:")
        for name, index in list(name_to_index.items())[:5]:
            print(f"{name} -> {index}")
        print("Exemplo de matriz de similaridade:")
        print(score_matrix[:5, :5])
    except FileNotFoundError:
        print(f"Arquivo {file_path} não encontrado.")