Explicacion general

Convencion de archivos:
- Los archivos SIN el sufijo "_2" no consideran las reglas del automata para la busqueda de reglas de asociacion.
- Los archivos CON el sufijo "_2" si consideran las reglas del automata (columnas r1..r27) en la busqueda.

Escenarios considerados:
- Escenario 1: sin todas las variables (subset).
- Escenario 1: con todas las variables.
- Escenario 2: sin todas las variables (subset).
- Escenario 2: con todas las variables.

Archivos .zip:
- dataset.zip contiene el dataset.
- Resultados 1.zip contiene los resultados sin todas las variables.
- Resultados 2.zip contiene los resultados con todas las variables.

Notas:
- CA_dataset.txt es el generador del dataset (en Mathematica).

Como ejecutar

Compilar y ejecutar (ejemplos):
1) MOEAD2:
   g++ -std=c++17 MOEAD2.cpp generar_individuo.cpp -o MOEAD2
   .\MOEAD2

2) MOEAD3:
   g++ -std=c++17 MOEAD3.cpp generar_individuo.cpp -o MOEAD3
   .\MOEAD3

3) Versiones con automata (sufijo _2):
   g++ -std=c++17 MOEAD2_2.cpp generar_individuo_2.cpp -o MOEAD2_2
   .\MOEAD2_2

   g++ -std=c++17 MOEAD3_2.cpp generar_individuo_2.cpp -o MOEAD3_2
   .\MOEAD3_2

Nota: ajusta los nombres si tus archivos difieren.
