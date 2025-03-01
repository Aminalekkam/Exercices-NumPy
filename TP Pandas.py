#Exercice 1: Creating and Modifying Series

import pandas as pd

# Création du dictionnaire
data = {'a': 100, 'b': 200, 'c': 300}

# Création de la Series
series = pd.Series(data)

# Affichage de la Series
print(series)


#Exercice 2 : Creating DataFrames

import pandas as pd

# Création du DataFrame initial
data = {'A': [1, 4, 7], 'B': [2, 5, 8], 'C': [3, 6, 9]}
df = pd.DataFrame(data)

# Ajout de la colonne D
df['D'] = [10, 11, 12]

# Suppression de la colonne B
df = df.drop(columns=['B'])

# Affichage du DataFrame final
print(df)


#Exercice 3:  DataFrame Indexing and Selection
    
import pandas as pd

# Création du DataFrame initial
data = {'A': [1, 4, 7], 'B': [2, 5, 8], 'C': [3, 6, 9]}
df = pd.DataFrame(data)

# Ajout de la colonne D
df['D'] = [10, 11, 12]

# Suppression de la colonne B
df = df.drop(columns=['B'])

# Affichage du DataFrame après suppression de B
print("DataFrame après suppression de la colonne B :")
print(df)

# Sélection de la colonne B (avant suppression)
df = pd.DataFrame(data)  # Recréer le DataFrame initial
column_B = df['B']
print("\nSélection de la colonne B :")
print(column_B)

# Sélection des colonnes A et C
columns_AC = df[['A', 'C']]
print("\nSélection des colonnes A et C :")
print(columns_AC)

# Sélection de la ligne avec l'index 1 en utilisant .loc
row_index_1 = df.loc[1]
print("\nSélection de la ligne avec index 1 :")
print(row_index_1)


#Exercice 4:  Adding and Removing DataFrame Elements

import pandas as pd
import numpy as np

# Création du DataFrame initial
data = {'A': [1, 4, 7], 'B': [2, 5, 8], 'C': [3, 6, 9]}
df = pd.DataFrame(data)

# Ajout de la colonne 'Sum' (somme des colonnes A, B et C)
df['Sum'] = df['A'] + df['B'] + df['C']

# Affichage du DataFrame avec la colonne Sum
print("DataFrame avec la colonne 'Sum':")
print(df)

# Suppression de la colonne 'Sum'
df = df.drop(columns=['Sum'])

# Affichage du DataFrame après suppression de Sum
print("\nDataFrame après suppression de la colonne 'Sum':")
print(df)

# Ajout d'une colonne 'Random' avec des nombres aléatoires
df['Random'] = np.random.rand(len(df))

# Affichage du DataFrame final
print("\nDataFrame avec la colonne 'Random':")
print(df)

#Exercice 5: Merging DataFrames
    
import pandas as pd

# Création du premier DataFrame (left)
left = pd.DataFrame({
    'key': [1, 2, 3],
    'A': ['A1', 'A2', 'A3'],
    'B': ['B1', 'B2', 'B3']
})

# Création du second DataFrame (right)
right = pd.DataFrame({
    'key': [1, 2, 3],
    'C': ['C1', 'C2', 'C3'],
    'D': ['D1', 'D2', 'D3']
})

# Fusion avec un inner join (par défaut)
merged_inner = pd.merge(left, right, on='key')
print("Fusion avec INNER JOIN :")
print(merged_inner)

# Fusion avec un outer join
merged_outer = pd.merge(left, right, on='key', how='outer')
print("\nFusion avec OUTER JOIN :")
print(merged_outer)

# Ajout d'une nouvelle colonne 'E' au second DataFrame
right['E'] = ['E1', 'E2', 'E3']

# Fusion mise à jour avec la nouvelle colonne
merged_with_E = pd.merge(left, right, on='key', how='outer')
print("\nFusion avec OUTER JOIN et ajout de la colonne 'E' :")
print(merged_with_E)


#Exercice 6: Data Cleaning


import pandas as pd
import numpy as np

# Création du DataFrame avec des valeurs NaN
data = {'A': [1.0, np.nan, 7.0], 'B': [np.nan, 5.0, 8.0], 'C': [3.0, 6.0, np.nan]}
df = pd.DataFrame(data)

# Remplacement des NaN par 0
df_filled_0 = df.fillna(0)
print("DataFrame après remplacement des NaN par 0 :")
print(df_filled_0)

# Remplacement des NaN par la moyenne de chaque colonne
df_filled_mean = df.fillna(df.mean())
print("\nDataFrame après remplacement des NaN par la moyenne de la colonne :")
print(df_filled_mean)

# Suppression des lignes contenant au moins un NaN
df_dropped = df.dropna()
print("\nDataFrame après suppression des lignes contenant des NaN :")
print(df_dropped)


#Exercice 7: Grouping and Aggregation

import pandas as pd

# Création du DataFrame
data = {'Category': ['A', 'B', 'A', 'B', 'A', 'B'], 
        'Value': [1, 2, 3, 4, 5, 6]}
df = pd.DataFrame(data)

# Groupement par 'Category' et calcul de la moyenne
mean_values = df.groupby('Category')['Value'].mean()
print("Moyenne des valeurs par catégorie :")
print(mean_values)

# Groupement par 'Category' et calcul de la somme
sum_values = df.groupby('Category')['Value'].sum()
print("\nSomme des valeurs par catégorie :")
print(sum_values)

# Groupement par 'Category' et comptage du nombre d'entrées
count_values = df.groupby('Category')['Value'].count()
print("\nNombre d'entrées par catégorie :")
print(count_values)


#Exercice 8:  Pivot Tables

import pandas as pd

# Création du DataFrame
data = {'Category': ['A', 'A', 'A', 'B', 'B', 'B'], 
        'Type': ['X', 'Y', 'X', 'Y', 'X', 'Y'], 
        'Value': [1, 2, 3, 4, 5, 6]}

df = pd.DataFrame(data)

# Création d'un pivot table avec la moyenne
pivot_mean = df.pivot_table(values='Value', index='Category', columns='Type', aggfunc='mean')
print("Pivot Table - Moyenne des valeurs :")
print(pivot_mean)

# Modification pour afficher la somme
pivot_sum = df.pivot_table(values='Value', index='Category', columns='Type', aggfunc='sum')
print("\nPivot Table - Somme des valeurs :")
print(pivot_sum)

# Ajout des marges (total par catégorie et type)
pivot_with_margins = df.pivot_table(values='Value', index='Category', columns='Type', aggfunc='mean', margins=True)
print("\nPivot Table - Moyenne avec marges :")
print(pivot_with_margins)

#Exercice 9: Time Series Data

import pandas as pd
import numpy as np

# Création d'une plage de dates
date_range = pd.date_range(start='2023-01-01', periods=6, freq='D')

# Génération de valeurs aléatoires
data = {'Date': date_range, 'Value': np.random.randint(1, 100, size=6)}

# Création du DataFrame
df = pd.DataFrame(data)

# Définition de la colonne 'Date' comme index
df.set_index('Date', inplace=True)

# Affichage du DataFrame initial
print("DataFrame original :")
print(df)

# Resampling pour calculer la somme sur des périodes de 2 jours
df_resampled = df.resample('2D').sum()

# Affichage du DataFrame après resampling
print("\nDataFrame après resampling (somme par période de 2 jours) :")
print(df_resampled)


#Exercice 10: Handling Missing Data

import pandas as pd
import numpy as np

# Création du DataFrame avec des NaN
data = {'A': [1.0, 2.0, np.nan], 'B': [np.nan, 5.0, 8.0], 'C': [3.0, np.nan, 9.0]}
df = pd.DataFrame(data)

# Affichage du DataFrame initial
print("DataFrame initial :")
print(df)

# Interpolation des valeurs manquantes
df_interpolated = df.interpolate()
print("\nDataFrame après interpolation des valeurs manquantes :")
print(df_interpolated)

# Suppression des lignes avec des NaN
df_dropped = df.dropna()
print("\nDataFrame après suppression des lignes avec des NaN :")
print(df_dropped)

#Exercice 11: DataFrame Operations

import pandas as pd

# Création du DataFrame
data = {'A': [1, 4, 7], 'B': [2, 5, 8], 'C': [3, 6, 9]}
df = pd.DataFrame(data)

# Affichage du DataFrame initial
print("DataFrame initial :")
print(df)

# Somme cumulative de chaque colonne
cumulative_sum = df.cumsum()
print("\nSomme cumulative :")
print(cumulative_sum)

# Produit cumulative de chaque colonne
cumulative_product = df.cumprod()
print("\nProduit cumulative :")
print(cumulative_product)

# Application de la fonction pour soustraire 1 de chaque élément
df_subtracted = df.applymap(lambda x: x - 1)
print("\nDataFrame après soustraction de 1 à chaque élément :")
print(df_subtracted)


    
    
    
    
    
    
    
    
    
    
    
    