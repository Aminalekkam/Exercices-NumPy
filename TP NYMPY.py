#EXO1 

import numpy as np

# 1D NumPy array
array_1d = np.array([5, 10, 15, 20, 25], dtype=np.float64)
print("1D Array:", array_1d)

# 2D NumPy array
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("2D Array Shape:", array_2d.shape)
print("2D Array Size:", array_2d.size)

# 3D NumPy array with random values
array_3d = np.random.rand(2, 3, 4)
print("3D Array Dimensions:", array_3d.ndim)
print("3D Array Shape:", array_3d.shape)

#EXO2 

import numpy as np

# 1D NumPy array from 0 to 9
array_1d = np.arange(10)
reversed_array = array_1d[::-1]
print("Reversed 1D Array:", reversed_array)

# 2D NumPy array (3x4) from 0 to 11
array_2d = np.arange(12).reshape(3, 4)
subarray = array_2d[:2, -2:]  # First two rows, last two columns
print("Original 2D Array:\n", array_2d)
print("Extracted Subarray:\n", subarray)

# 2D NumPy array (5x5) with random integers between 0 and 10
array_random = np.random.randint(0, 11, (5, 5))
array_random[array_random > 5] = 0  # Replace elements greater than 5 with 0
print("Modified 5x5 Array:\n", array_random)


#EXO3


import numpy as np

# 1. Création d'une matrice identité 3x3
identity_matrix = np.eye(3)
print("Identity Matrix:\n", identity_matrix)

# Affichage des attributs
print("ndim:", identity_matrix.ndim)  # Nombre de dimensions
print("shape:", identity_matrix.shape)  # Forme du tableau
print("size:", identity_matrix.size)  # Nombre total d'éléments
print("itemsize:", identity_matrix.itemsize, "bytes")  # Taille d'un élément en bytes
print("nbytes:", identity_matrix.nbytes, "bytes")  # Taille totale en bytes

# 2. Tableau de 10 nombres espacés uniformément entre 0 et 5
evenly_spaced = np.linspace(0, 5, 10)
print("\nEvenly Spaced Array:", evenly_spaced)
print("Datatype:", evenly_spaced.dtype)

# 3. Création d'un tableau 3D (2,3,4) avec des valeurs aléatoires suivant une distribution normale
random_3d_array = np.random.randn(2, 3, 4)  # Distribution normale (moyenne 0, variance 1)
print("\n3D Array:\n", random_3d_array)
print("Sum of elements:", np.sum(random_3d_array))


#EXO4 

import numpy as np

# 1. Création d'un tableau 1D de 20 entiers aléatoires entre 0 et 50
array_1d = np.random.randint(0, 50, size=20)
print("1D Array:\n", array_1d)

# Extraction des éléments aux indices [2, 5, 7, 10, 15]
fancy_indexing_result = array_1d[[2, 5, 7, 10, 15]]
print("Fancy Indexing Result:\n", fancy_indexing_result)

# 2. Création d'un tableau 2D (4x5) avec des entiers aléatoires entre 0 et 30
array_2d = np.random.randint(0, 30, size=(4, 5))
print("\n2D Array:\n", array_2d)

# Sélection des éléments supérieurs à 15
mask_greater_than_15 = array_2d[array_2d > 15]
print("Elements Greater Than 15:\n", mask_greater_than_15)

# 3. Création d'un tableau 1D de 10 entiers aléatoires entre -10 et 10
array_with_negatives = np.random.randint(-10, 10, size=10)
print("\nOriginal 1D Array with Negatives:\n", array_with_negatives)

# Remplacement des valeurs négatives par 0
array_with_negatives[array_with_negatives < 0] = 0
print("Modified Array (Negatives Set to Zero):\n", array_with_negatives)


#EXO5 

import numpy as np

# 1. Création de deux tableaux 1D de longueur 5 avec des entiers aléatoires entre 0 et 10
array1 = np.random.randint(0, 10, size=5)
array2 = np.random.randint(0, 10, size=5)

# Concaténation des deux tableaux
concatenated_array = np.concatenate([array1, array2])

print("Array 1:", array1)
print("Array 2:", array2)
print("Concatenated Array:", concatenated_array)

# 2. Création d'un tableau 2D (6x4) avec des entiers aléatoires entre 0 et 10
array_2d = np.random.randint(0, 10, size=(6, 4))

# Division du tableau en deux parties égales selon l'axe des lignes
split_arrays_row = np.split(array_2d, 2, axis=0)

print("\nOriginal 2D Array (6x4):\n", array_2d)
print("First Split (Top 3 rows):\n", split_arrays_row[0])
print("Second Split (Bottom 3 rows):\n", split_arrays_row[1])

# 3. Création d'un tableau 2D (3x6) avec des entiers aléatoires entre 0 et 10
array_2d_col = np.random.randint(0, 10, size=(3, 6))

# Division du tableau en trois parties égales selon l'axe des colonnes
split_arrays_col = np.split(array_2d_col, 3, axis=1)

print("\nOriginal 2D Array (3x6):\n", array_2d_col)
print("First Split (First 2 columns):\n", split_arrays_col[0])
print("Second Split (Middle 2 columns):\n", split_arrays_col[1])
print("Third Split (Last 2 columns):\n", split_arrays_col[2])


#EXO6


import numpy as np

# 1. Création d'un tableau 1D de 15 entiers aléatoires entre 1 et 100
array_1d = np.random.randint(1, 100, size=15)

# Calcul des statistiques
mean_value = np.mean(array_1d)
median_value = np.median(array_1d)
std_dev = np.std(array_1d)
variance = np.var(array_1d)

print("1D Array:", array_1d)
print("Mean:", mean_value)
print("Median:", median_value)
print("Standard Deviation:", std_dev)
print("Variance:", variance)

# 2. Création d'un tableau 2D (4x4) avec des entiers aléatoires entre 1 et 50
array_2d = np.random.randint(1, 50, size=(4, 4))

# Somme des lignes et colonnes
sum_rows = np.sum(array_2d, axis=1)  # Somme des lignes
sum_columns = np.sum(array_2d, axis=0)  # Somme des colonnes

print("\n2D Array (4x4):\n", array_2d)
print("Sum of each row:", sum_rows)
print("Sum of each column:", sum_columns)

# 3. Création d'un tableau 3D (2x3x4) avec des entiers aléatoires entre 1 et 20
array_3d = np.random.randint(1, 20, size=(2, 3, 4))

# Recherche des valeurs max et min selon chaque axe
max_axis0 = np.max(array_3d, axis=0)  # Max sur l'axe 0
max_axis1 = np.max(array_3d, axis=1)  # Max sur l'axe 1
max_axis2 = np.max(array_3d, axis=2)  # Max sur l'axe 2

min_axis0 = np.min(array_3d, axis=0)  # Min sur l'axe 0
min_axis1 = np.min(array_3d, axis=1)  # Min sur l'axe 1
min_axis2 = np.min(array_3d, axis=2)  # Min sur l'axe 2

print("\n3D Array (2x3x4):\n", array_3d)
print("Max along axis 0:\n", max_axis0)
print("Max along axis 1:\n", max_axis1)
print("Max along axis 2:\n", max_axis2)

print("\nMin along axis 0:\n", min_axis0)
print("Min along axis 1:\n", min_axis1)
print("Min along axis 2:\n", min_axis2)


#EXO7

import numpy as np

# 1. Create a 1D NumPy array with the numbers from 1 to 12. Reshape it to (3, 4)
array_1d = np.arange(1, 13)
array_2d = array_1d.reshape(3, 4)
print("Reshaped 2D Array (3, 4):\n", array_2d)

# 2. Create a 2D NumPy array with random integers between 1 and 10, of shape (3, 4)
random_array = np.random.randint(1, 11, size=(3, 4))
transposed_array = random_array.T  # Transpose the array
print("\nTransposed Array:\n", transposed_array)

# 3. Create a 2D NumPy array of shape (2, 3) with random integers between 1 and 10, and flatten it
flattened_array = np.random.randint(1, 11, size=(2, 3)).flatten()
print("\nFlattened Array (1D):\n", flattened_array)


#EXO8

import numpy as np

# 1. Create a 2D NumPy array of shape (3, 4) with random integers between 1 and 10.
array_2d = np.random.randint(1, 11, size=(3, 4))

# Subtract the mean of each column from the respective column elements
mean_per_column = array_2d.mean(axis=0)
result = array_2d - mean_per_column
print("Result after subtracting column means:\n", result)

# 2. Create two 1D NumPy arrays of length 4 with random integers between 1 and 5.
array_1d_a = np.random.randint(1, 6, size=4)
array_1d_b = np.random.randint(1, 6, size=4)

# Compute the outer product using broadcasting
outer_product = np.outer(array_1d_a, array_1d_b)
print("\nOuter product of the two arrays:\n", outer_product)

# 3. Create a 2D NumPy array of shape (4, 5) with random integers between 1 and 10.
array_2d_2 = np.random.randint(1, 11, size=(4, 5))

# Add 10 to all elements of the array that are greater than 5
array_2d_2[array_2d_2 > 5] += 10
print("\nModified array (add 10 to elements > 5):\n", array_2d_2)


#EXO9

import numpy as np

# 1. Create a 1D NumPy array with random integers between 1 and 20 of size 10.
array_1d = np.random.randint(1, 21, size=10)

# Sort the array in ascending order and print the result
sorted_array = np.sort(array_1d)
print("Sorted 1D Array:\n", sorted_array)

# 2. Create a 2D NumPy array of shape (3, 5) with random integers between 1 and 50.
array_2d = np.random.randint(1, 51, size=(3, 5))

# Sort the array by the second column (axis=0 is for sorting columns)
sorted_array_2d = array_2d[array_2d[:, 1].argsort()]
print("\nSorted 2D Array by second column:\n", sorted_array_2d)

# 3. Create a 1D NumPy array with random integers between 1 and 100 of size 15.
array_1d_2 = np.random.randint(1, 101, size=15)

# Find the indices of all elements greater than 50
indices_greater_than_50 = np.where(array_1d_2 > 50)[0]
print("\nIndices of elements greater than 50:\n", indices_greater_than_50)

#EXO10

import numpy as np

# 1. Create a 2D NumPy array of shape (2, 2) with random integers between 1 and 10.
array_2d_1 = np.random.randint(1, 11, size=(2, 2))

# Compute and print the determinant of the array
determinant = np.linalg.det(array_2d_1)
print("Determinant of the 2D array:\n", determinant)

# 2. Create a 2D NumPy array of shape (3, 3) with random integers between 1 and 5.
array_2d_2 = np.random.randint(1, 6, size=(3, 3))

# Compute and print the eigenvalues and eigenvectors of the array
eigenvalues, eigenvectors = np.linalg.eig(array_2d_2)
print("\nEigenvalues of the 3x3 array:\n", eigenvalues)
print("Eigenvectors of the 3x3 array:\n", eigenvectors)

# 3. Create two 2D NumPy arrays of shape (2, 3) and (3, 2) with random integers between 1 and 10.
array_2d_3 = np.random.randint(1, 11, size=(2, 3))
array_2d_4 = np.random.randint(1, 11, size=(3, 2))

# Compute and print the matrix product of the two arrays
matrix_product = np.dot(array_2d_3, array_2d_4)
print("\nMatrix product of the two arrays:\n", matrix_product)


#EXO11

import numpy as np
import matplotlib.pyplot as plt

# 1. Create a 1D NumPy array of 10 random samples from a uniform distribution over [0, 1)
uniform_array = np.random.uniform(0, 1, size=10)
print("1D Array from uniform distribution:\n", uniform_array)

# 2. Create a 2D NumPy array of shape (3, 3) with random samples from a normal distribution with mean 0 and standard deviation 1.
normal_array = np.random.normal(0, 1, size=(3, 3))
print("\n2D Array from normal distribution (mean=0, std=1):\n", normal_array)

# 3. Create a 1D NumPy array of 20 random integers between 1 and 100.
random_integers = np.random.randint(1, 101, size=20)

# Compute and print the histogram of the array with 5 bins
plt.hist(random_integers, bins=5, edgecolor='black')
plt.title('Histogram of Random Integers')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()


#EXO12

import numpy as np

# 1. Create a 2D NumPy array of shape (5, 5) with random integers between 1 and 20.
array_2d_1 = np.random.randint(1, 21, size=(5, 5))

# Select and print the diagonal elements of the array
diagonal_elements = np.diagonal(array_2d_1)
print("Diagonal elements:\n", diagonal_elements)

# 2. Create a 1D NumPy array of 10 random integers between 1 and 50.
array_1d = np.random.randint(1, 51, size=10)

# Function to check if a number is prime
def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(np.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

# Use advanced indexing to select and print all prime numbers
prime_elements = array_1d[np.vectorize(is_prime)(array_1d)]
print("\nPrime numbers from the 1D array:\n", prime_elements)

# 3. Create a 2D NumPy array of shape (4, 4) with random integers between 1 and 10.
array_2d_2 = np.random.randint(1, 11, size=(4, 4))

# Select and print all even elements
even_elements = array_2d_2[array_2d_2 % 2 == 0]
print("\nEven elements from the 2D array:\n", even_elements)


#EXO13

import numpy as np

# 1. Create a 1D NumPy array of length 10 with random integers between 1 and 10. Introduce np.nan at random positions.
array_1d = np.random.randint(1, 11, size=10).astype(float)  # Convert to float to allow np.nan

# Introduce np.nan at random positions
nan_indices = np.random.choice(array_1d.shape[0], size=3, replace=False)  # Choose 3 random indices for NaN
array_1d[nan_indices] = np.nan
print("1D Array with np.nan at random positions:\n", array_1d)

# 2. Create a 2D NumPy array of shape (3, 4) with random integers between 1 and 10. Replace all elements less than 5 with np.nan.
array_2d = np.random.randint(1, 11, size=(3, 4)).astype(float)  # Convert to float to allow np.nan

# Replace all elements less than 5 with np.nan
array_2d[array_2d < 5] = np.nan
print("\n2D Array with elements less than 5 replaced with np.nan:\n", array_2d)

# 3. Create a 1D NumPy array of length 15 with random integers between 1 and 20. Identify and print the indices of all np.nan values.
array_1d_2 = np.random.randint(1, 21, size=15).astype(float)  # Convert to float to allow np.nan

# Introduce np.nan at random positions in the 1D array
nan_indices_2 = np.random.choice(array_1d_2.shape[0], size=4, replace=False)  # Choose 4 random indices for NaN
array_1d_2[nan_indices_2] = np.nan

# Find and print the indices of all np.nan values
nan_indices_in_array = np.where(np.isnan(array_1d_2))[0]
print("\nIndices of np.nan values in the 1D array:\n", nan_indices_in_array)


#EXO14

import numpy as np
import time

# 1. Create a large 1D NumPy array with 1 million random integers between 1 and 100.
array_1d_large = np.random.randint(1, 101, size=1_000_000)

# Measure the time taken to compute the mean and standard deviation
start_time = time.time()
mean = np.mean(array_1d_large)
std_dev = np.std(array_1d_large)
end_time = time.time()
print(f"Mean: {mean}, Standard Deviation: {std_dev}")
print(f"Time taken for mean and std computation: {end_time - start_time} seconds\n")

# 2. Create two large 2D NumPy arrays of shape (1000, 1000) with random integers between 1 and 10.
array_2d_1 = np.random.randint(1, 11, size=(1000, 1000))
array_2d_2 = np.random.randint(1, 11, size=(1000, 1000))

# Measure the time taken to perform element-wise addition
start_time = time.time()
result_addition = array_2d_1 + array_2d_2
end_time = time.time()
print(f"Time taken for element-wise addition: {end_time - start_time} seconds\n")

# 3. Create a 3D NumPy array of shape (100, 100, 100) with random integers between 1 and 10.
array_3d = np.random.randint(1, 11, size=(100, 100, 100))

# Measure the time taken to compute the sum along each axis
start_time = time.time()
sum_along_axis_0 = np.sum(array_3d, axis=0)
sum_along_axis_1 = np.sum(array_3d, axis=1)
sum_along_axis_2 = np.sum(array_3d, axis=2)
end_time = time.time()
print(f"Time taken for sum along each axis: {end_time - start_time} seconds")


#EXO15

import numpy as np

# 1. Create a 1D NumPy array with the numbers from 1 to 10. Compute and print the cumulative sum and cumulative product of the array.
array_1d = np.arange(1, 11)

cumulative_sum = np.cumsum(array_1d)
cumulative_product = np.cumprod(array_1d)

print(f"Cumulative Sum of the array:\n{cumulative_sum}")
print(f"Cumulative Product of the array:\n{cumulative_product}\n")

# 2. Create a 2D NumPy array of shape (4, 4) with random integers between 1 and 20. Compute and print the cumulative sum along the rows and the columns.
array_2d = np.random.randint(1, 21, size=(4, 4))

cumulative_sum_rows = np.cumsum(array_2d, axis=1)  # Cumulative sum along rows
cumulative_sum_columns = np.cumsum(array_2d, axis=0)  # Cumulative sum along columns

print(f"Cumulative Sum along rows:\n{cumulative_sum_rows}")
print(f"Cumulative Sum along columns:\n{cumulative_sum_columns}\n")

# 3. Create a 1D NumPy array with 10 random integers between 1 and 50. Compute and print the minimum, maximum, and sum of the array.
array_1d_2 = np.random.randint(1, 51, size=10)

min_value = np.min(array_1d_2)
max_value = np.max(array_1d_2)
sum_value = np.sum(array_1d_2)

print(f"Minimum value: {min_value}")
print(f"Maximum value: {max_value}")
print(f"Sum of the array: {sum_value}")


#EXO16

import numpy as np

# 1. Create an array of 10 dates starting from today with a daily frequency and print the array.
today = np.datetime64('today', 'D')  # Get today's date
dates_daily = np.arange(today, today + np.timedelta64(10, 'D'), dtype='datetime64[D]')  # Daily frequency
print(f"Array of 10 dates starting from today:\n{dates_daily}\n")

# 2. Create an array of 5 dates starting from January 1, 2022 with a monthly frequency and print the array.
start_date = np.datetime64('2022-01-01', 'D')  # Starting date: January 1, 2022
# Instead of using np.timedelta64, we directly use np.datetime64 with a monthly step
dates_monthly = np.datetime64('2022-01-01', 'M') + np.arange(0, 5) * np.timedelta64(1, 'M')
print(f"Array of 5 dates starting from January 1, 2022 with monthly frequency:\n{dates_monthly}\n")

# 3. Create a 1D array with 10 random timestamps in the year 2023. Convert the timestamps to NumPy datetime64 objects and print the result.
random_timestamps = np.random.randint(0, 365, size=10)  # Random day of 2023 (0-364)
timestamps_2023 = np.datetime64('2023-01-01') + random_timestamps.astype('timedelta64[D]')  # Convert to datetime64
print(f"Array of 10 random timestamps in the year 2023:\n{timestamps_2023}")


#EXO17

import numpy as np

# 1. Create a 1D NumPy array of length 5 with custom data type to store integers and their corresponding binary representation as strings. Print the array.
# Define a custom data type with two fields: 'integer' and 'binary'.
dtype = np.dtype([('integer', np.int32), ('binary', 'U32')])  # 'U32' for a string of 32 characters

# Create an array with this custom dtype and initialize with integers and their binary representation.
arr1 = np.array([(1, bin(1)[2:]), (2, bin(2)[2:]), (3, bin(3)[2:]), (4, bin(4)[2:]), (5, bin(5)[2:])], dtype=dtype)

print(f"Array of integers and their binary representations:\n{arr1}\n")

# 2. Create a 2D NumPy array of shape (3, 3) with a custom data type to store complex numbers. Initialize the array with some complex numbers and print the array.
# Define a custom data type for complex numbers.
dtype_complex = np.dtype([('real', np.float64), ('imag', np.float64)])

# Initialize the array with complex numbers.
arr2 = np.array([[(1, 2), (3, 4), (5, 6)], [(7, 8), (9, 10), (11, 12)], [(13, 14), (15, 16), (17, 18)]], dtype=dtype_complex)

print(f"Array of complex numbers:\n{arr2}\n")

# 3. Create a structured array to store information about books with fields: title (string), author (string), and pages (integer). Add information for three books and print the structured array.
# Define a custom data type for the book information.
dtype_books = np.dtype([('title', 'U50'), ('author', 'U50'), ('pages', np.int32)])

# Initialize the structured array with book information.
books = np.array([('Book1', 'Author1', 100), ('Book2', 'Author2', 200), ('Book3', 'Author3', 300)], dtype=dtype_books)

print(f"Structured array with book information:\n{books}")

















