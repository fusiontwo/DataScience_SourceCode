
'''
1. minkowski distance for r=1, 2, ...
2. Cosine similarity and dot product
3. most similar k

'''

import numpy as np


def euclidean_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1) - np.array(vector2))  # norm은 벡터의 크기를 계산



def minkowski_distance(vector1, vector2, p):
    if p == np.inf:
        return np.max(np.abs(np.array(vector1) - np.array(vector2)))
    else:
        return np.sum(np.abs(np.array(vector1) - np.array(vector2))**p)**(1/p)



def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    vector1_len = np.linalg.norm(vector1)  # 각각을 unit vector로 미리 계산해두면 시간 줄어듬.
    vector2_len = np.linalg.norm(vector2)
    
    similarity = dot_product / (vector1_len * vector2_len)
    
    return similarity



def dot_product(vector1, vector2):
    return np.dot(vector1, vector2)   



def topk_vectors(one_vector, vectors, k):  # vectors는 이차원 벡터

    similarities = [cosine_similarity(one_vector, v) for v in vectors]

    # argsort() : sorting한 인덱스 값을 리턴
    topk_indices = np.argsort(similarities)[-k:][::-1]  # 뒤에서 k개를 slicing, 순서는 뒤에서 부터 읽으므로 가장 큰 값이 앞에 옴.  
    # 순서를 먼저 뒤집고 k개를 slicing 하면 효율성이 위의 코드보다 떨어짐.
    #topk_indices = np.argsort(similarities)[::-1][:k]
    #[start:stop:step]

    topk_vectors = vectors[topk_indices]
    
    print(topk_vectors)
    


if __name__ == '__main__':

    dim = 10

    vector1 = np.random.randint(0, 100, dim)
    vector2 = np.random.randint(0, 100, dim)
    
    print(vector1)
    print(vector2)
    
    print(f"""
          norm1(v1,v2) = {minkowski_distance(vector1, vector2, 1)}
          norm2(v1,v2) = {minkowski_distance(vector1, vector2, 2)}
          norm_max(v1, v2) = {minkowski_distance(vector1, vector2, np.inf)}
          """)
 
    num_vectors = 1000000
    vectors = np.random.randint(0, 101, (num_vectors, dim))
    
    topk_vectors(vector1, vectors, k=3)
    
    