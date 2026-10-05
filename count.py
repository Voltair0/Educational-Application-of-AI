with open('winequality-white.csv', 'r', encoding='utf-8') as f:
    numar_linii = sum(1 for row in f)

numar_instante = numar_linii - 1

print(f"Numărul de instanțe este: {numar_instante}")