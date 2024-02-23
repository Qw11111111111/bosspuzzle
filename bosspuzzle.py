import numpy as np

def initialising(size):
  Matrix = np.arange(size ** 2)
  np.random.shuffle(Matrix)
  return Matrix.reshape((size, size))

def _print(Matrix):
  print(" S", end = " ")
  [print(*[i + 1 for i in range(len(Matrix))], end = " ")]
  print("")
  print("Z ", end = " ")
  [print(*["-" for i in range(len(Matrix))])]
  [print(str(i + 1) + " ",*Matrix[i], sep = "|") for i in range(len(Matrix))]

def _input(Matrix):
  Test = False
  while Test == False:
    Zeile = int(input("Welches Feld wollen Sie verschieben [Zeile]?\n")) - 1
    Spalte = int(input("Welches Feld wollen Sie verschieben [Spalte]?\n")) - 1

    Nullelement = np.argwhere(Matrix == 0)[0]
    if (Spalte == Nullelement[1] or Spalte == Nullelement[1] + 1 or Spalte == Nullelement[1] - 1) and ( Zeile == Nullelement[0] or Zeile == Nullelement[0] + 1 or Zeile == Nullelement[0] - 1):
      Test = True
  return [Zeile, Spalte, Nullelement]

def update(Matrix, Feld):
  Zeile, Spalte, Nullelement = Feld[0], Feld[1], Feld[2]
  Matrix[Nullelement[0]][Nullelement[1]] = Matrix[Zeile, Spalte]
  Matrix[Zeile, Spalte] = 0
  return Matrix
  
def auswertung(Matrix, size):
  c = 0
  solution = np.arange(size ** 2).reshape(size, size)
  for i in range(size):
    for j in range(size):
      if Matrix[i][j] == solution[i][j]:
        c += 1
    if c == size ** 2:
      print("Sie haben das Puzzle vollendet")
      return "Gewonnen"

def main():
  mode = str(input("Hallo zum Boss Puzzle um es zu beginnnen drücken Sie [enter]"))
  size = int(input("wie gross soll das spielfeld sein\n"))
  Matrix = initialising(size)
  while True:
    _print(Matrix)
    Feld = _input(Matrix)
    Matrix = update(Matrix, Feld)
    if mode == "auto":
      Matrix = np.arange(size ** 2).reshape(size, size)
    if auswertung(Matrix, size) == "Gewonnen":
      break
  
  
if __name__ == "__main__":
  main()