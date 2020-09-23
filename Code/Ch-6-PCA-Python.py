from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

# define a small 5×3 matrix
matrix = array([[5, 6,3],
                [8, 10,2],
                [12, 18,4],
                [8, 10,1],
                [6, 12,5],])
print("original Matrix: ")
print(matrix)

# step-1 calculate the mean of each column
Mean_col = mean(matrix, axis=0)# [7.8,11.2 3]
print("Mean of each column: ")
print(Mean_col)

# center columns by subtracting column means
normalized_data = matrix - Mean_col
print("Normalized Data: ")
print(normalized_data)

# calculate covariance matrix of centered matrix
cov_matrix = cov(normalized_data.T)
print("Covariance Matrix",cov_matrix)

# eigendecomposition of covariance matrix
values, vectors = eig(cov_matrix)
print("Eigen vectors: ",vectors)
print("Eigen values: ",values)

# project data on the new axes
projected_data = vectors.T.dot(normalized_data.T) # (3x3)(3x5)->(3x5)
print("projected data",projected_data.T) # (5x3)