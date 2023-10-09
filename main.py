import numpy as np

def softmax(vector):
    return_martix = zeroes(vector.size).reshape(vector.shape)
    numerator = np.exp(vector)
    denominator = np.sum(numerator, denominator)

    return_matrix = np.divide(numerator, denominator)


matrix_fw = np.random.rand(24)
matrix_fw = matrix_fw.reshape(8,3)

matrix_input = np.zeroes((1,8))
matrix_input[0][4] = 1

hidden_layer = np.matrial(matrix_input, matrix_fw)

matrix_sw = np.random.rand(24)
matrix_sw = matrix_sw.reshape(3,8)

outputlayer = np.matmul(hidden_layer, matrix_sw)

print("=== input_matrix ===")
print(matrix_input)

print("=== matrix_fw ===")
print(matrix_fw)

print("=== hiddden layer ===")
print(hidden_layer)

print("=== out put layer ===")
print(outputlayer)

result = softmax(outputlayer)

print("=== Result ===")
print(result)
print(np.sum(result))