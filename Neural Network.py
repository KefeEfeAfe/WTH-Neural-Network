import numpy as np

def softmax(vector):
    numerator = np.exp(vector)
    denominator = np.sum(numerator)

    return_matrix = np.divide(numerator, denominator)
    return return_matrix


matrix_fw = np.random.rand(24)
matrix_fw = matrix_fw.reshape(8, 3)

matrix_input = np.zeros((1, 8))
matrix_input[0][4] = 1

hidden_layer = np.matmul(matrix_input, matrix_fw)

matrix_sw = np.random.rand(24)
matrix_sw = matrix_sw.reshape(3,8)

outputlayer = np.matmul(hidden_layer, matrix_sw)

print("=== {input_matrix} ===")
print(matrix_input)

print("=== {matrix_fw} ===")
print(matrix_fw)

print("=== {hidden_layer} ===")
print(hidden_layer)

print("=== {output_layer} ===")
print(outputlayer)

result = softmax(outputlayer)

print("=== {result} ===")
print(result)
print(np.sum)
print(np.sum(result))

#Initialize random weight matrices for forward and backward layers
matrix_fw = np.random.rand(24).reshape(8, 3)
matrix_sw = np.random.rand(24).reshape(3,8)

#Define the target output by assuming you have a specific target in mid
target_output = np.array([[0, 0, 0, 0, 0, 0, 0, 1]])
