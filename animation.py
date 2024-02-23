import numpy as np
import time
from queue import PriorityQueue
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from boss2 import scramble, solve_puzzle, heuristic, get_neighbors, IDA_star



def animation(size):

        global cycle_A, cycle_IDA, path_A, path_IDA, restart_A, restart_IDA
        cycle_A, cycle_IDA = -1, -1
        initial_matrix = goal_matrix
        restart_IDA, restart_A = False, False
        


        def data_gen(*args):
            
            global cycle_A, cycle_IDA, path_IDA, path_A, restart_A, restart_IDA, initial_matrix

            #####init####
            if cycle_A == -1 and cycle_IDA == -1 and restart_A is False and restart_IDA is False:
                if size <= 3:
                    time.sleep(3)
                initial_matrix = scramble(size)
                restart_A = True
                restart_IDA = True   
                pass

            
            
            
            #####A_star#####
            

            if cycle_A == -1 and restart_A == True:
                        path_A = solve_puzzle(initial_matrix, goal_matrix, size)
                        mat_A.set_data(initial_matrix)
                        cycle_A += 1
                        pass

            elif cycle_A == len(path_A) or (cycle_A == -1 and restart_A == False):
                        cycle_A = -1
                        restart_A = False
                        pass
                    
            elif cycle_A < len(path_A):
                        matrix_A = path_A[cycle_A]
                        mat_A.set_data(matrix_A)
                        cycle_A += 1
                        pass
                        
                        
                        
            ####IDA_star####
                    

            if cycle_IDA == -1 and restart_IDA == True:
                        path_IDA = IDA_star(initial_matrix, goal_matrix, size)
                        mat_IDA.set_data(initial_matrix)
                        cycle_IDA += 1
                        pass

            elif cycle_IDA == len(path_IDA) or (cycle_IDA == -1 and restart_IDA == False):
                        cycle_IDA = -1
                        restart_IDA = False
                        pass
                    
            elif cycle_IDA < len(path_IDA):
                        matrix_IDA = path_IDA[cycle_IDA]
                        mat_IDA.set_data(matrix_IDA)
                        cycle_IDA += 1
                        pass
            
            return None


       
        fig, ax = plt.subplots()
        plt.title("A*  |  IDA*")
        ax_A = fig.add_subplot(1, 2, 1)
    
        ax_IDA = fig.add_subplot(1, 2, 2)
        
        
        mat_A = ax_A.matshow(initial_matrix)
        mat_IDA = ax_IDA.matshow(initial_matrix)
        mat_A.set_label("A*")
        ani = FuncAnimation(fig, data_gen, frames = None, interval = 300, blit = False, init_func=lambda *args: None, cache_frame_data=False)       
        plt.show()

def path_zu_matrix(i,matrix, path):
    Nullelement = np.argwhere(matrix == 0)[0]
    matrix_new = matrix.copy()
    matrix_new[Nullelement[0] + path[i][0]][Nullelement[1] + path[i][1]] = 0 
    matrix_new[Nullelement[0]][Nullelement[1]] = matrix[Nullelement[0] + path[i][0]][Nullelement[1] + path[i][1]]
    return matrix_new

def solve_puzzle(initial_state, goal_state, size):
    frontier = PriorityQueue()
    explored = set()

    frontier.put((0, initial_state, []))  # Priority, state, path

    while not frontier.empty():
        _, current_state_tuple, path = frontier.get()
        current_state = np.array(current_state_tuple)
        if np.array_equal(current_state, goal_state):
            matrix = [initial_state]
            for i in range(len(path)):
                matrix.append(path_zu_matrix(i, matrix[-1], path))
            return matrix
    
        current_state_tuple = tuple(map(tuple, current_state))
        if current_state_tuple not in explored:
            explored.add(current_state_tuple)
            neighbors = get_neighbors(current_state, size)
            for neighbor, move in neighbors:
                new_path = path + [move]
                priority = len(new_path) + heuristic(neighbor, goal_state)
                #print(priority)
                frontier.put((priority, tuple(map(tuple, neighbor)), new_path))

    return None

def main():
    global goal_tuple
    global goal_matrix
    size = int(input("size\n"))
    goal_matrix = np.arange(size ** 2).reshape((size, size))
    goal_tuple = tuple(map(tuple, goal_matrix))
    animation(size)

if __name__ == "__main__":
    main()