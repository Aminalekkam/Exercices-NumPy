import numpy as np
import matplotlib.pyplot as plt


#Exo1

# Création de l'array x avec 100 points entre -10 et 10
x = np.linspace(-10, 10, 100)

# Définition du polynôme y = 2x^3 - 5x^2 + 3x - 7
y = 2*x**3 - 5*x**2 + 3*x - 7

# Création de la figure avec taille 10x6 pouces
plt.figure(figsize=(10, 6))

# Tracé de la courbe en bleu
plt.plot(x, y, 'b', label=r'$y = 2x^3 - 5x^2 + 3x - 7$')

# Ajout des labels et du titre
plt.xlabel('x')
plt.ylabel('y')
plt.title('Tracé du polynôme $y = 2x^3 - 5x^2 + 3x - 7$')

# Ajout d'une légende
plt.legend()

# Affichage du graphique
plt.show()


#EXO2

import numpy as np
import matplotlib.pyplot as plt

# Création de l'array x avec 500 points entre 0.1 et 10
x = np.linspace(0.1, 10, 500)

# Calcul des fonctions exponentielle et logarithmique
y1 = np.exp(x)  # Exponentielle
y2 = np.log(x)  # Logarithme naturel

# Création de la figure
plt.figure(figsize=(10, 6))

# Tracé des courbes
plt.plot(x, y1, 'r--', label=r'$y = e^x$')  # Exponentielle en rouge pointillé
plt.plot(x, y2, 'b-', label=r'$y = \ln(x)$')  # Logarithme en bleu ligne pleine

# Ajout des labels et du titre
plt.xlabel('x')
plt.ylabel('y')
plt.title('Tracé des fonctions exponentielle et logarithmique')

# Ajout d'une grille et d'une légende
plt.grid(True)
plt.legend()

# Sauvegarde du graphique en PNG avec 100 DPI
plt.savefig('exponential_log_plot.png', dpi=100)

# Affichage du graphique
plt.show()

#EXO3

import numpy as np
import matplotlib.pyplot as plt

# Création des valeurs pour les subplots
x1 = np.linspace(-2 * np.pi, 2 * np.pi, 500)  # Pour tan et arctan
x2 = np.linspace(-2, 2, 500)  # Pour sinh et cosh

# Calcul des fonctions
y_tan = np.tan(x1)
y_arctan = np.arctan(x1)
y_sinh = np.sinh(x2)
y_cosh = np.cosh(x2)

# Création de la figure avec deux subplots en une seule ligne
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Premier subplot : tan(x) et arctan(x)
axes[0].plot(x1, y_tan, 'r--', label=r'$\tan(x)$')  # Rouge pointillé
axes[0].plot(x1, y_arctan, 'b-', label=r'$\arctan(x)$')  # Bleu ligne pleine
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title(r'Tracé de $\tan(x)$ et $\arctan(x)$')  # Correction du format LaTeX
axes[0].legend()
axes[0].grid(True)
axes[0].set_ylim(-10, 10)  # Limiter l'axe Y pour éviter les grandes valeurs de tan(x)

# Deuxième subplot : sinh(x) et cosh(x)
axes[1].plot(x2, y_sinh, 'g-', label=r'$\sinh(x)$')  # Vert ligne pleine
axes[1].plot(x2, y_cosh, 'm--', label=r'$\cosh(x)$')  # Magenta pointillé
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title(r'Tracé de $\sinh(x)$ et $\cosh(x)$')  # Correction du format LaTeX
axes[1].legend()
axes[1].grid(True)

# Ajustement de l'affichage pour éviter le chevauchement
plt.tight_layout()

# Génération des données pour l'histogramme
n = np.random.randn(1000)  # 1000 valeurs d'une distribution normale

# Création d'une nouvelle figure pour l'histogramme
plt.figure(figsize=(8, 5))

# Tracé de l'histogramme avec 30 bins
plt.hist(n, bins=30, color='c', edgecolor='k', alpha=0.7)

# Personnalisation
plt.title('Histogramme d’une distribution normale')
plt.xlabel('Valeurs')
plt.ylabel('Fréquence')
plt.grid(True)

# Définition des limites de l'axe X
plt.xlim(n.min(), n.max())

# Affichage des graphiques
plt.show()


#EXO4

import numpy as np
import matplotlib.pyplot as plt

# Génération des données
np.random.seed(42)  # Pour des résultats reproductibles
x = np.random.uniform(-10, 10, 500)  # 500 valeurs uniformément réparties entre -10 et 10
y = np.sin(x) + np.random.normal(0, 0.2, 500)  # y = sin(x) avec du bruit aléatoire

# Définition des tailles et couleurs des marqueurs en fonction des valeurs de y
sizes = 50 * (np.abs(y) + 0.5)  # La taille dépend de |y| (pour éviter des points trop petits)
colors = y  # La couleur dépend des valeurs de y

# Création de la figure
plt.figure(figsize=(10, 6))

# Tracé du scatter plot
plt.scatter(x, y, s=sizes, c=colors, cmap='coolwarm', alpha=0.75, edgecolors='k')

# Personnalisation
plt.title("Scatter Plot de y = sin(x) avec bruit")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)

# Suppression des ticks des axes
plt.xticks([])
plt.yticks([])

# Sauvegarde du graphique en PDF
plt.savefig("scatter_plot.pdf", format="pdf", bbox_inches="tight")

# Affichage du graphique
plt.show()


#EXO5


import numpy as np
import matplotlib.pyplot as plt

# Génération des données
x = np.linspace(-5, 5, 200)  # 200 points entre -5 et 5
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)  # Création de la grille 2D

# Définition de la fonction f(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))  # f(x, y) = sin(sqrt(x^2 + y^2))

# Création de la figure
plt.figure(figsize=(8, 6))

# Tracé des courbes de niveau (contour plot)
contour = plt.contour(X, Y, Z, levels=20, cmap='plasma')  # 20 niveaux avec la colormap "plasma"
plt.clabel(contour, inline=True, fontsize=8)  # Ajout des labels sur les contours

# Personnalisation
plt.title("Contour Plot de $f(x, y) = \sin(\sqrt{x^2 + y^2})$")
plt.xlabel("x")
plt.ylabel("y")

# Affichage du graphique
plt.show()


#EXO6 

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Génération des données
x = np.arange(-5, 5.25, 0.25)  # Valeurs de -5 à 5 avec un pas de 0.25
y = np.arange(-5, 5.25, 0.25)
X, Y = np.meshgrid(x, y)  # Création de la grille 2D

# Définition de la fonction Z = cos(sqrt(X^2 + Y^2))
Z = np.cos(np.sqrt(X**2 + Y**2))

# Création de la figure et d'un axe 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Tracé du 3D surface plot
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')

# Personnalisation
ax.set_title("3D Surface Plot de $Z = \cos(\sqrt{X^2 + Y^2})$")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Modification de l'angle de vue
ax.view_init(elev=30, azim=135)  # Élévation à 30° et azimut à 135°

# Ajout d'une barre de couleur
fig.colorbar(surf, shrink=0.6, aspect=10)

# Affichage du graphique
plt.show()

#EXO7 


import numpy as np
import matplotlib.pyplot as plt

# Génération des données
x = np.linspace(-2, 2, 10)  # 10 points entre -2 et 2
y1 = x**2
y2 = x**3
y3 = x**4

# Création de la figure
plt.figure(figsize=(8, 6))

# Tracé des trois fonctions avec différents styles
plt.plot(x, y1, 'ro--', label=r'$y = x^2$')   # Rouge, cercles, ligne en pointillés
plt.plot(x, y2, 'bs-.', label=r'$y = x^3$')   # Bleu, carrés, ligne en tirets-points
plt.plot(x, y3, 'gD:', label=r'$y = x^4$')    # Vert, losanges, ligne en pointillés

# Personnalisation
plt.xlabel("x")
plt.ylabel("y")
plt.title("Tracé de différentes fonctions avec styles variés")
plt.legend()  # Ajout de la légende
plt.grid(True)  # Ajout de la grille

# Affichage du graphique
plt.show()


#EXO8

import numpy as np
import matplotlib.pyplot as plt

# Génération des données
x = np.linspace(1, 100, 50)  # 50 points entre 1 et 100
y1 = 2**x  # Exponentielle en base 2
y2 = np.log2(x)  # Logarithme base 2

# Création de la figure
plt.figure(figsize=(12, 6))

# Tracé des courbes
plt.plot(x, y1, 'r-', label=r'$y = 2^x$')  # Ligne rouge pour l'exponentielle
plt.plot(x, y2, 'b--', label=r'$y = \log_2(x)$')  # Ligne bleue en pointillés pour le logarithme

# Personnalisation
plt.yscale('log')  # Échelle logarithmique sur l'axe Y
plt.xlabel("x")
plt.ylabel("y (échelle logarithmique)")
plt.title("Tracé de $y = 2^x$ et $y = \log_2(x)$ avec une échelle logarithmique")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Grille pour meilleure lisibilité

# Affichage du graphique
plt.show()


#EXO9

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Génération des données
x = np.arange(-5, 5.25, 0.25)  # Valeurs de -5 à 5 avec un pas de 0.25
y = np.arange(-5, 5.25, 0.25)
X, Y = np.meshgrid(x, y)  # Création de la grille 2D

# Définition de la fonction Z = cos(sqrt(X^2 + Y^2))
Z = np.cos(np.sqrt(X**2 + Y**2))

# Création de la figure avec deux sous-graphiques
fig = plt.figure(figsize=(12, 6))

# Premier subplot avec un angle de vue
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')
ax1.set_title("Vue de dessus (elev=90, azim=0)")
ax1.view_init(elev=90, azim=0)  # Vue de dessus

# Deuxième subplot avec un autre angle de vue
ax2 = fig.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='none')
ax2.set_title("Vue isométrique (elev=30, azim=135)")
ax2.view_init(elev=30, azim=135)  # Vue isométrique

# Ajustement de l'affichage
plt.tight_layout()

# Affichage du graphique
plt.show()


#EXO10 


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Génération des données
x = np.arange(-5, 5.25, 0.25)  # Valeurs de -5 à 5 avec un pas de 0.25
y = np.arange(-5, 5.25, 0.25)
X, Y = np.meshgrid(x, y)  # Création de la grille 2D

# Définition de la fonction Z = sin(X) * cos(Y)
Z = np.sin(X) * np.cos(Y)

# Création de la figure et du sous-graphe 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Tracé du Wireframe
ax.plot_wireframe(X, Y, Z, color='black')

# Personnalisation
ax.set_title("3D Wireframe Plot de $Z = \sin(X) * \cos(Y)$")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Modification de l'angle de vue
ax.view_init(elev=30, azim=45)  # Élévation à 30° et azimut à 45°

# Affichage du graphique
plt.show()

#EXO11


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Génération des données
x = np.arange(-5, 5.25, 0.25)  # Valeurs de -5 à 5 avec un pas de 0.25
y = np.arange(-5, 5.25, 0.25)
X, Y = np.meshgrid(x, y)  # Création de la grille 2D

# Définition de la fonction Z = exp(-0.1 * (X^2 + Y^2))
Z = np.exp(-0.1 * (X**2 + Y**2))

# Création de la figure et du sous-graphe 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Tracé du 3D Contour Plot
contour = ax.contour3D(X, Y, Z, 50, cmap='viridis')  # 50 niveaux et colormap 'viridis'

# Personnalisation
ax.set_title("3D Contour Plot de $Z = e^{-0.1 \times (X^2 + Y^2)}$")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Affichage de la légende de la colormap
fig.colorbar(contour, ax=ax, shrink=0.5, aspect=5)

# Affichage du graphique
plt.show()


#EXO12


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Création de l'array t avec 100 points entre 0 et 2π
t = np.linspace(0, 2 * np.pi, 100)

# Calcul des coordonnées X, Y et Z
X = np.sin(t)
Y = np.cos(t)
Z = t

# Création de la figure et du sous-graphe 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Tracé du 3D Parametric Plot
ax.plot(X, Y, Z, color='magenta')  # Changement de couleur en magenta

# Personnalisation
ax.set_title("3D Parametric Plot de $X = sin(t), Y = cos(t), Z = t$")
ax.set_xlabel("X (sin(t))")
ax.set_ylabel("Y (cos(t))")
ax.set_zlabel("Z (t)")

# Affichage du graphique
plt.show()


#EXO13


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Création des arrays x et y avec 10 points espacés entre -5 et 5
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
X, Y = np.meshgrid(x, y)  # Création de la grille 2D

# Calcul des valeurs de z = exp(-0.1 * (x^2 + y^2))
Z = np.exp(-0.1 * (X**2 + Y**2))

# Création de la figure et du sous-graphe 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Tracé du 3D Bar Plot
bars = ax.bar3d(X.flatten(), Y.flatten(), np.zeros_like(Z.flatten()), 
                 1, 1, Z.flatten(), shade=True, cmap='viridis')

# Personnalisation
ax.set_title("3D Bar Plot de $Z = e^{-0.1 \times (X^2 + Y^2)}$")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Ajout de la barre de couleurs
fig.colorbar(bars, ax=ax, shrink=0.5, aspect=5)

# Changer l'angle de vue
ax.view_init(elev=30, azim=45)

# Affichage du graphique
plt.show()


#EXO14

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Création des arrays X, Y et Z avec des valeurs de -5 à 5
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
z = np.linspace(-5, 5, 10)

# Création des grilles 3D
X, Y, Z = np.meshgrid(x, y, z)

# Calcul des composantes du champ vectoriel
U = -Y
V = X
W = Z

# Création de la figure et du sous-graphe 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Calcul des couleurs en fonction de la norme des vecteurs
norm = np.linalg.norm([U, V, W], axis=0)  # Calcul de la norme des vecteurs
colors = plt.cm.coolwarm(norm / np.max(norm))  # Appliquer la colormap 'coolwarm'

# Extraire les couleurs RGB de la colormap
rgb_colors = colors[:, :3]  # On garde uniquement les 3 premières valeurs (RGB)

# Tracé du champ vectoriel 3D avec quiver
quiver = ax.quiver(X, Y, Z, U, V, W, length=0.5, color=rgb_colors)

# Personnalisation
ax.set_title("3D Vector Field de $U = -Y, V = X, W = Z$")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Ajout de la barre de couleur
fig.colorbar(quiver, ax=ax, shrink=0.5, aspect=5)

# Affichage du graphique
plt.show()


#EXO15

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Création des arrays x, y, z avec 100 valeurs tirées d'une distribution normale
x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
z = np.random.normal(0, 1, 100)

# Création de la figure et du sous-graphe 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Tracé du scatter plot 3D avec les points colorés selon les valeurs de z
scatter = ax.scatter(x, y, z, c=z, cmap='viridis')

# Personnalisation
ax.set_title("3D Scatter Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Ajout de la barre de couleur
fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

# Affichage du graphique
plt.show()


#EXO16


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Création de l'array t avec 100 points linéairement espacés entre 0 et 4π
t = np.linspace(0, 4 * np.pi, 100)

# Calcul des coordonnées X, Y, Z
X = np.sin(t)
Y = np.cos(t)
Z = t

# Création de la figure et du sous-graphe 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Tracé de la ligne 3D avec couleur et largeur de ligne personnalisées
ax.plot(X, Y, Z, color='r', linewidth=2)

# Personnalisation des axes et du titre
ax.set_title("3D Line Plot: $X = \sin(t), Y = \cos(t), Z = t$")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Affichage du graphique
plt.show()

#EXO17


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Création des arrays X et Y avec des valeurs allant de -5 à 5 avec un pas de 0.1
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)

# Création de la grille 2D
X, Y = np.meshgrid(x, y)

# Calcul de Z = sin(sqrt(X^2 + Y^2))
Z = np.sin(np.sqrt(X**2 + Y**2))

# Création de la figure et du sous-graphe 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Tracé du 3D filled contour plot
contour = ax.contourf(X, Y, Z, 50, cmap='plasma')

# Personnalisation des axes et du titre
ax.set_title("3D Filled Contour Plot: $Z = \sin(\sqrt{X^2 + Y^2})$")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Ajout de la barre de couleur
fig.colorbar(contour, ax=ax, shrink=0.5, aspect=5)

# Affichage du graphique
plt.show()

#EXO18


import numpy as np
import matplotlib.pyplot as plt

# Création des arrays x et y avec 50 points allant de -5 à 5
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)

# Création de la grille 2D
X, Y = np.meshgrid(x, y)

# Calcul de Z = sin(sqrt(x^2 + y^2))
Z = np.sin(np.sqrt(X**2 + Y**2))

# Création de la figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)

# Tracé de la heatmap
heatmap = ax.imshow(Z, extent=[-5, 5, -5, 5], origin='lower', cmap='plasma', interpolation='nearest')

# Ajout de la barre de couleur
fig.colorbar(heatmap, ax=ax)

# Personnalisation des axes et du titre
ax.set_title("3D Heatmap: $Z = \sin(\sqrt{x^2 + y^2})$")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Affichage du graphique
plt.show()


#EXO19


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Génération de 1000 points aléatoires en 3D
x = np.random.randn(1000)
y = np.random.randn(1000)
z = np.random.randn(1000)

# Création de la figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Création du graphique de densité 3D
hist, xedges, yedges = np.histogram2d(x, y, bins=30, range=[[-5, 5], [-5, 5]])

# Création d'une grille pour l'affichage
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = np.zeros_like(xpos)

# Les valeurs de densité (histogramme)
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

# Tracé des barres dans le graphique 3D
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='c', alpha=0.6)

# Personnalisation des axes et du titre
ax.set_title("3D Density Plot")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Density")

# Affichage du graphique
plt.show()








