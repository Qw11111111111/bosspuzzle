import numpy as np
import random
from queue import PriorityQueue
from bosspuzzle import initialising, update, _input, _print, auswertung
import time
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def scramble(size):
    Matrix = np.arange(size ** 2)
    Matrix = Matrix.reshape((size, size))
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    _matrix = Matrix.copy()
    __matrix = Matrix.copy()
    if size <= 4:
        d = 4
    else:
        d = 2
    for i in range(size ** d):    
        Nullelement = np.argwhere(_matrix == 0)[0]
        while True:
            move = moves[random.randint(0,3)]
            new_position = Nullelement + move
            if 0 <= new_position[0] < size and 0 <= new_position[1] < size:
                break
        _matrix[Nullelement[0]][Nullelement[1]] = __matrix[Nullelement[0] + move[0]][Nullelement[1] + move[1]]
        _matrix[Nullelement[0] + move[0]][Nullelement[1] + move[1]] = 0
        __matrix = _matrix.copy()
    if not np.array_equal(_matrix, Matrix):
        return _matrix

def animation(initial_matrix, path, search_mode):
        cycle = len(path)
        _print(initial_matrix)
        
        def data_gen(cycle):
            if search_mode == "a_star":
                global _matrix
                if cycle == len(path):
                    time.sleep(2)
                    return mat
                    
                if cycle == 0:
                    _matrix = initial_matrix.copy()
                Nullelement = np.argwhere(_matrix == 0)[0]
                plt.title(f"Boss_Puzzle aktion {cycle}")
                matrix_new = _matrix.copy()
                _cycle = cycle
                new_row = Nullelement[0] + path[_cycle][0]
                new_col = Nullelement[1] + path[_cycle][1] 

                if not (0 <= new_row < _matrix.shape[0] and 0 <= new_col < _matrix.shape[1]):
                    print("Index out of bounds!")
                    return mat

                matrix_new[Nullelement[0] + path[_cycle][0]][Nullelement[1] + path[_cycle][1]] = 0 #null wird auf dieses feld verschoben
                matrix_new[Nullelement[0]][Nullelement[1]] = _matrix[Nullelement[0] + path[_cycle][0]][Nullelement[1] + path[_cycle][1]] # null wird von diesem feld verschoben
                _matrix = matrix_new
                mat.set_data(matrix_new)
                return mat

            elif search_mode == "IDA_star":
                if cycle == 0:
                    _matrix = initial_matrix.copy()
                    mat.set_data(_matrix)
                    return mat
                
                if cycle == len(path):
                    return mat
                
                plt.title(f"Boss_Puzzle aktion {cycle}")
                mat.set_data(path[cycle])
                return mat
                


        fig, ax = plt.subplots()
        mat = ax.matshow(initial_matrix)
        plt.colorbar(mat)
        ani = FuncAnimation(fig, data_gen, cycle + 1, interval = 100, blit = False, init_func=lambda *args: None)      
        plt.show()

def heuristic(state, goal):
    return np.sum(np.abs(state - goal))

def possible_Moves(current, size):
    current = np.array(current)
    zero_position = np.argwhere(current == 0)[0]
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    Possible_Moves = set()
    for move in moves:
        new_position = zero_position + move
        if 0 <= new_position[0] < size and 0 <= new_position[1] < size:
            new = current.copy()
            new[zero_position[0], zero_position[1]] = current[new_position[0], new_position[1]]
            new[new_position[0], new_position[1]] = 0
            Possible_Moves.add(tuple(map(tuple,new)))
    return list(Possible_Moves)

def walking_distance(matrix):
    pass

def IDA_star(initial_matrix, goal_matrix, size):

    def search(_path, is_in_path, g, threshold):
        current = _path[-1]
        newf = g + heuristic(np.array(current), goal_matrix)
        if newf > threshold:
            return newf
        if current == goal_tuple:
            return FOUND
        minimum = np.inf
        
        for move in possible_Moves(current, size):
            if move in is_in_path:
                continue
            _path.append(move)
            is_in_path.add(move)
            t = search(_path = _path, is_in_path = is_in_path, g = g + 1, threshold = threshold)
            if t == FOUND:
                current = _path[-1]
                return FOUND
            elif t < minimum:
                minimum = t
            _path.pop()
            is_in_path.remove(move)
        return minimum

    threshold = heuristic(initial_matrix, goal_matrix)
    goal_tuple = tuple(map(tuple, goal_matrix))
    _initial_matrix = tuple(map(tuple, initial_matrix))
    path = [_initial_matrix]
    in_path = set()
    FOUND = object()
    while True:
        t = search(path, in_path, 0, threshold)
        if t == FOUND:            
            return path
        elif t is np.inf:
            return []
        else:
            threshold = t

def get_neighbors(matrix, size):
    neighbors = []
    zero_position = np.argwhere(matrix == 0)[0]
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]  

    for move in moves:
        new_position = zero_position + move
        if 0 <= new_position[0] < size and 0 <= new_position[1] < size:
            neighbor = matrix.copy()
            neighbor[zero_position[0], zero_position[1]] = matrix[new_position[0], new_position[1]]
            neighbor[new_position[0], new_position[1]] = 0
            neighbors.append((neighbor, move))
    return neighbors

def solve_puzzle(initial_state, goal_state, size):
    frontier = PriorityQueue()
    explored = set()

    frontier.put((heuristic(initial_state, goal_state), 0, initial_state, []))

    while not frontier.empty():
        f_score, g_score, current_state_tuple, path = frontier.get()
        current_state = np.array(current_state_tuple)
        if np.array_equal(current_state, goal_state):
            return path
    
        current_state_tuple = tuple(map(tuple, current_state))
        if current_state_tuple not in explored:
            explored.add(current_state_tuple)
            neighbors = get_neighbors(current_state, size)
            for neighbor, move in neighbors:
                new_path = path + [move]
                g_score = len(new_path)
                f_score = len(new_path) + heuristic(neighbor, goal_state)
                frontier.put((f_score, g_score, tuple(map(tuple, neighbor)), new_path))

    return None 

def main():
    global goal_tuple
    mode = str(input("Hallo zum Boss Puzzle. Um es zu beginnen, drücken Sie [enter]. Um einen algorithums zu nutzen drücken Sie [auto]\n"))
    size = int(input("Wie groß soll das Spielfeld sein?\n"))
    initial_matrix = initialising(size)
    if size > 2:
        initial_matrix = scramble(size)
    goal_matrix = np.arange(size ** 2).reshape(size, size)
    goal_tuple = tuple(map(tuple, goal_matrix))

    if mode == "auto":
        _print(initial_matrix)
        _start = time.perf_counter()
        __solution_path = IDA_star(initial_matrix, goal_matrix, size)
        _end = time.perf_counter()
        start = time.perf_counter()
        _solution_path = solve_puzzle(initial_matrix, goal_matrix, size)
        end = time.perf_counter()
        if __solution_path is not None:
            solution_path = []
            for i in __solution_path:
                solution_path.append(1)

            solution_path = np.array(__solution_path)

            print("IDA_star steps: ", len(solution_path) - 1)
            print(f"time to compute: {end - start}")
            print("A_star steps: ", len(_solution_path) - 1)
            print(f"time to compute: {_end - _start}")

            animation(initial_matrix, __solution_path, search_mode = "IDA_star")
        else:
            print("Keine Lösung gefunden.")
            print(end - start)

    else:
        Matrix = initial_matrix
        while True:
            _print(Matrix)
            Feld = _input(Matrix)
            Matrix = update(Matrix, Feld)
            if auswertung(Matrix, size) == "Gewonnen":
                break

if __name__ == "__main__":
    main()