import numpy as np
import copy
import random
import time
import math
from collections import defaultdict
# board = np.zeros((11,11), dtype=int)
# print(board)

import pygame as pg

def draw_dash_line(surface, color, start, end, width=1, dash_length=4):
    x1, y1 = start
    x2, y2 = end
    dl = dash_length

    if (x1 == x2):
        ycoords = [y for y in range(y1, y2, dl if y1 < y2 else -dl)]
        xcoords = [x1] * len(ycoords)
    elif (y1 == y2):
        xcoords = [x for x in range(x1, x2, dl if x1 < x2 else -dl)]
        ycoords = [y1] * len(xcoords)
    else:
        a = abs(x2 - x1)
        b = abs(y2 - y1)
        c = round(math.sqrt(a**2 + b**2))
        dx = dl * a / c
        dy = dl * b / c

        xcoords = [x for x in numpy.arange(x1, x2, dx if x1 < x2 else -dx)]
        ycoords = [y for y in numpy.arange(y1, y2, dy if y1 < y2 else -dy)]

    next_coords = list(zip(xcoords[1::2], ycoords[1::2]))
    last_coords = list(zip(xcoords[0::2], ycoords[0::2]))
    for (x1, y1), (x2, y2) in zip(next_coords, last_coords):
        start = (round(x1), round(y1))
        end = (round(x2), round(y2))
        pg.draw.line(surface, color, start, end, width)


####################################################################################################################
# create the initial empty chess board in the game window
def draw_board():
    
    global xbline, w_size, pad, sep
    
    xbline = bline + 8                        # Add 4 extra line on each boundaries to make chains of 5 that cross boundaries easier to see
    w_size = 720                              # window size
    pad = 36                                  # padding size
    sep = int((w_size-pad*2)/(xbline-1))      # separation between lines = [window size (720) - padding*2 (36*2)]/(Total lines (19) -1)
    
    surface = pg.display.set_mode((w_size, w_size))
    pg.display.set_caption("Gomuku (a.k.a Five-in-a-Row)")
    
    color_line = [0, 0, 0]
    color_board = [241, 196, 15]

    surface.fill(color_board)
    
    for i in range(0, xbline):
        draw_dash_line(surface, color_line, [pad, pad+i*sep], [w_size-pad, pad+i*sep])
        draw_dash_line(surface, color_line, [pad+i*sep, pad], [pad+i*sep, w_size-pad])
        
    for i in range(0, bline):
        pg.draw.line(surface, color_line, [pad+4*sep, pad+(i+4)*sep], [w_size-pad-4*sep, pad+(i+4)*sep], 4)
        pg.draw.line(surface, color_line, [pad+(i+4)*sep, pad+4*sep], [pad+(i+4)*sep, w_size-pad-4*sep], 4)

    pg.display.update()
    
    return surface


####################################################################################################################
# Draw the stones on the board at pos = [row, col]. 
# Draw a black circle at pos if color = 1, and white circle at pos if color =  -1
# row and col are be the indices on the 11x11 board array
# dark gray and light gray circles are also drawn on the dotted grid to indicate a phantom stone piece
def draw_stone(surface, pos, color=0):

    color_black = [0, 0, 0]
    color_dark_gray = [75, 75, 75]
    color_white = [255, 255, 255]
    color_light_gray = [235, 235, 235]
    
    matx = pos[0] + 4 + bline*np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).flatten()
    matx1 = np.logical_and(matx >= 0, matx < xbline)
    maty = pos[1] + 4 + bline*np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]).T.flatten()
    maty1 = np.logical_and(maty >= 0, maty < xbline)
    mat = np.logical_and(np.logical_and(matx1, maty1), np.array([[True, True, True], [True, False, True], [True, True, True]]).flatten())

    if color==1:
        pg.draw.circle(surface, color_black, [pad+(pos[0]+4)*sep, pad+(pos[1]+4)*sep], 15, 0)
        for f, x, y in zip(mat, matx, maty):
            if f:
                pg.draw.circle(surface, color_dark_gray, [pad+x*sep, pad+y*sep], 15, 0)
                
    elif color==-1:
        pg.draw.circle(surface, color_white, [pad+(pos[0]+4)*sep, pad+(pos[1]+4)*sep], 15, 0)
        for f, x, y in zip(mat, matx, maty):
            if f:
                pg.draw.circle(surface, color_light_gray, [pad+x*sep, pad+y*sep], 15, 0)
        
    pg.display.update()
    

####################################################################################################################
def print_winner(surface, winner=0):
    if winner == 2:
        msg = "Draw! So White wins"
        color = [170,170,170]
    elif winner == 1:
        msg = "Black wins!"
        color = [0,0,0]
    elif winner == -1:
        msg = 'White wins!'
        color = [255,255,255]
    else:
        return
        
    font = pg.font.Font('freesansbold.ttf', 32)
    text = font.render(msg, True, color)
    textRect = text.get_rect()
    textRect.topleft = (0, 0)
    surface.blit(text, textRect)
    pg.display.update()

####################################################################################################################

class zmtBoard(object):
    """
    board for game
    """
 
    def __init__(self, width=11, height=11, n_in_row=5):
        self.width = width
        self.height = height 
        self.states = {} # ��¼��ǰ���̵�״̬������λ�ã�ֵ�����ӣ��������������ʾ��������
        self.last_change = {"last":-1}
        self.last_last_change = {"last_last":-1}
        self.n_in_row = n_in_row # ��ʾ������ͬ����������һ������ʤ��
        self.steps = 0
 
    def init_board(self):
        self.availables = list(range(self.width * self.height)) # ��ʾ���������кϷ���λ�ã�����򵥵���Ϊ�յ�λ�ü��Ϸ�
 
        for m in self.availables:
            self.states[m] = 0 # 0��ʾ��ǰλ��Ϊ��
 
    def move_to_location(self, move):
        h = move  // self.width
        w = move  %  self.width
        return [h, w]
 
    def location_to_move(self, location):
        if(len(location) != 2):
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if(move not in range(self.width * self.height)):
            return -1
        return move
 
    def update(self, player, move): # player��move�����ӣ���������
        self.states[move] = player
        self.availables.remove(move)
        self.last_last_change["last_last"] = self.last_change["last"]
        self.last_change["last"] = move
        self.steps += 1

class zmtMCTS(object):
    """
    AI player, use Monte Carlo Tree Search with UCB
    """
 
    def __init__(self, board, play_turn, n_in_row=5, time=5, max_actions=3000):
 
        self.board = board
        self.play_turn = play_turn # ����˳��
        self.calculation_time = float(time) # �������ʱ��
        self.max_actions = max_actions # ÿ��ģ��Ծ������еĲ���
        self.n_in_row = n_in_row
 
        self.player = play_turn[0] # �ֵ����Գ��֣����Գ���˳���е�һ�����ǵ���
        self.confident = 1.96 # UCB�еĳ��� 1.96
        self.plays = {} # ��¼�ŷ�����ģ��Ĵ�����������(player, move)��������ң����ӣ�
        self.wins = {} # ��¼�ŷ���ʤ�Ĵ���
        self.max_depth = 1
        self.skip = False
 
    def get_action(self): # return move
 
        if len(self.board.availables) == 1:
            return self.board.availables[0] # ����ֻʣ���һ������λ�ã�ֱ�ӷ���
 
        # ÿ�μ�����һ��ʱ��Ҫ���plays��wins����Ϊ����AI����ҵ�2����֮���������̵ľ��淢���˱仯��ԭ���ļ�¼�Ѿ��������ˡ���ԭ����ͨ��һ�����ڿ�������ʤ��һ�����������գ���Ӱ�����ڵĽ����������һ������û��ô����ʤ����
        self.plays = {} 
        self.wins = {}
        self.skip = False
        simulations = 0
        begin = time.time()
        while time.time() - begin < self.calculation_time:
            board_copy = copy.deepcopy(self.board)  # ģ����޸�board�Ĳ��������Ա�������������ԭboard���и���
            play_turn_copy = copy.deepcopy(self.play_turn) # ÿ��ģ�ⶼ���밴�չ̶���˳����У����Խ��������ֹ˳���޸�
            self.run_simulation(board_copy, play_turn_copy) # ����MCTS
            simulations += 1
 
        print("total simulations=", simulations)
        
        self.skip = self.skipf(self.board)
        move = self.select_one_move(self.board) # ѡ������ŷ�
        
        location = self.board.move_to_location(move)
        print('Maximum depth searched:', self.max_depth)
 
        print("AI move: %d,%d\n" % (location[0], location[1]))
 
        return move
 
    def run_simulation(self, board, play_turn):
        """
        MCTS main process
        """
 
        plays = self.plays
        wins = self.wins
        availables = board.availables
 
        player = self.get_player(play_turn) # ��ȡ��ǰ���ֵ����
        visited_states = set() # ��¼��ǰ·���ϵ�ȫ���ŷ�
        winner = 0
        expand = True
 
        # Simulation
        for t in range(1, self.max_actions + 1):
            # Selection
            # ��������ŷ�����ͳ����Ϣ�����ȡUCB�����ŷ�

            if all(plays.get((player, move)) for move in availables):
                log_total = math.log(
                    sum(plays[(player, move)] for move in availables))
                value, move = max(
                    ((wins[(player, move)] / plays[(player, move)]) +
                     math.sqrt(self.confident * log_total / plays[(player, move)]), move)
                    for move in availables)
            
            else:
                adjacents = []
                if len(availables) > self.n_in_row:
                    adjacents = self.adjacent_moves(board, player, plays) # û��ͳ����Ϣ���ڽ�λ��
                
                if len(adjacents):
                    move = random.choice(adjacents)
                else:
                    peripherals = []
                    for move in availables:
                        if not plays.get((player, move)):
                            peripherals.append(move) # û��ͳ����Ϣ����Χλ��
                    move = random.choice(peripherals) 
            
            board.update(player, move)
 
            # Expand
            # ÿ��ģ�������չһ�Σ�ÿ����չֻ����һ���ŷ�
            if expand and (player, move) not in plays:
                expand = False
                plays[(player, move)] = 0
                wins[(player, move)] = 0
                if t > self.max_depth:
                    self.max_depth = t
 
            visited_states.add((player, move))
 
            is_full = not len(availables)
            win = self.check_winner(board) != 0
            
            winner=0
            if win:
                winner = self.check_winner(board)
            
            if is_full or win: # ��Ϸ������û������λ�û�����һ�ʤ
                break
 
            player = self.get_player(play_turn)
 
        # Back-propagation
        for player, move in visited_states:
            if (player, move) not in plays:
                continue
            plays[(player, move)] += 1 # ��ǰ·���������ŷ���ģ�������1
            if player == winner:
                wins[(player, move)] += 1 # ��ʤ��ҵ������ŷ���ʤ��������1
 
    def get_player(self, players):
        p = players.pop(0)
        players.append(p)
        return p
 
    def select_one_move(self, board):
        
        if self.skip and board.steps >3:
            percent_wins, move = max(
                (self.wins.get((self.player, move), 0) /
                self.plays.get((self.player, move), 1),
                move)
                for move in self.skip)
        
        else:
            limited = self.adjacent2(board)+self.adjacent3(board)
            percent_wins, move = max(
                (self.wins.get((self.player, move), 0) /
                 self.plays.get((self.player, move), 1),
                 move)
                for move in limited) # ѡ��ʤ����ߵ��ŷ� # self.board.availables
        
        return move
    
    def check_winner(self, board):
        """
        ����Ƿ�����һ�ʤ
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)

        array11 = np.concatenate((array_2d[-4:,-4:], array_2d[-4:,:],array_2d[-4:,:4]), axis=1)
        array12 = np.concatenate((array_2d[:,-4:], array_2d,array_2d[:,:4]), axis=1)
        array13 = np.concatenate((array_2d[:4,-4:], array_2d[:4,:],array_2d[:4,:4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19��19 �ж�����

        n=board.height
        
        tent=board.last_change["last"]
        i=board.move_to_location(tent)[0]
        j=board.move_to_location(tent)[1]

        indexlist1=list([i+4,j+k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A1=list(board1[m[0],m[1]] for m in indexlist1)
        count = 0
        for num in A1:
            if num == array_2d[i][j]:
                count += 1
                if count == 5:
                    return array_2d[i][j]
            else:
                count = 0

        indexlist2=list([i+k+4,j+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A2=list(board1[m[0],m[1]] for m in indexlist2)
        count = 0
        for num in A2:
            if num == array_2d[i][j]:
                count += 1
                if count == 5:
                    return array_2d[i][j]
            else:
                count = 0

        indexlist3=list([i+k+4,j+k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A3=list(board1[m[0],m[1]] for m in indexlist3)
        count = 0
        for num in A3:
            if num == array_2d[i][j]:
                count += 1
                if count == 5:
                    return array_2d[i][j]
            else:
                count = 0

        indexlist4=list([i-k+4,j+k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A4=list(board1[m[0],m[1]] for m in indexlist4)
        count = 0
        for num in A4:
            if num == array_2d[i][j]:
                count += 1
                if count == 5:
                    return array_2d[i][j]
            else:
                count = 0

        for a in range(n):
            for b in range(n):
                if array_2d[a][b] == 0:
                    return 0
        return 2

        # moved = list(set(range(board.width * board.height)) - set(board.availables))
        # if(len(moved) < self.n_in_row + 2):
        #     return 0
        # 
        # width = board.width
        # height = board.height
        # states = board.states
        # n = self.n_in_row
        # count = 0
        # for m in moved:
        #     count += 1
        #     h = m // width
        #     w = m % width
        #     player = states[m]
        # 
        #     if (w in range(width - n + 1) and
        #         len(set(states[i] for i in range(m, m + n))) == 1): # ��������һ��
        #         return player
        # 
        #     if (h in range(height - n + 1) and
        #         len(set(states[i] for i in range(m, m + n * width, width))) == 1): # ��������һ��
        #         return player
        # 
        #     if (w in range(width - n + 1) and h in range(height - n + 1) and
        #         len(set(states[i] for i in range(m, m + n * (width + 1), width + 1))) == 1): # ��б��������һ��
        #         return player
        # 
        #     if (w in range(n - 1, width) and h in range(height - n + 1) and
        #         len(set(states[i] for i in range(m, m + n * (width - 1), width - 1))) == 1): # ��б��������һ��
        #         return player
        # 
        # if count == width*height:
        #     return 2
        # 
        # return 0
    
    def adjacent_moves(self, board, player, plays):
        """
        ��ȡ��ǰ������������ӵ��ڽ�λ����û��ͳ����Ϣ��λ��
        """
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        adjacents = set()
        width = board.width
        height = board.height
     
        for m in moved:
            h = m // width
            w = m % width
            if w < width - 1:
                adjacents.add(m + 1) # ��
            if w == width - 1:
                adjacents.add(m + 1 - width) # �ҵ���  
            if w > 0:
                adjacents.add(m - 1) # ��
            if w == 0:
                adjacents.add(m - 1 + width) # ����
            if h < height - 1:
                adjacents.add(m + width) # ��
            if h == height - 1:
                adjacents.add(m + width - height*width) # �µ���
            if h > 0:
                adjacents.add(m - width) # ��
            if h == 0:
                adjacents.add(m - width + height*width) # �ϵ���
            if w < width - 1 and h < height - 1:
                adjacents.add(m + width + 1) # ����
            if w == width - 1 and h == height - 1:
                adjacents.add(m + width + 1 - width - height*width) # ���µ�����
            if w > 0 and h < height - 1:
                adjacents.add(m + width - 1) # ����
            if w == 0 and h == height - 1:
                adjacents.add(m + width - 1 + width - height*width) # ���µ�����
            if w < width - 1 and h > 0:
                adjacents.add(m - width + 1) # ����
            if w == width - 1 and h == 0:
                adjacents.add(m - width + 1 - width + height*width) # ���ϵ�����
            if w > 0 and h > 0:
                adjacents.add(m - width - 1) # ����
            if w == 0 and h == 0:
                adjacents.add(m - width - 1 + width + height*width) # ���ϵ�����
     
        adjacents = list(set(adjacents) - set(moved))
        for move in adjacents:
            if plays.get((player, move)):
                adjacents.remove(move)
        return adjacents

    def checkp4(self, board):
        """
        ������4
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # ֵ���� -1,0,1
    
        array11 = np.concatenate((array_2d[-4:,-4:], array_2d[-4:,:],array_2d[-4:,:4]), axis=1)
        array12 = np.concatenate((array_2d[:,-4:], array_2d,array_2d[:,:4]), axis=1)
        array13 = np.concatenate((array_2d[:4,-4:], array_2d[:4,:],array_2d[:4,:4]), axis=1)
    
        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19��19 �ж����� ֵ���� -1,0,1
        
        n=board.height
        
        tent=board.last_change["last"]
        i=board.move_to_location(tent)[0]
        j=board.move_to_location(tent)[1]
        
        player=array_2d[i][j]
        
        target410=[0, player, player, player, player]
        target411=[player, player, player, player, 0]
        target42=[player,player,0,player,player]
        target430=[player,0,player,player,player]
        target431=[player,player,player,0,player]
        
        window_size=len(target410)
        results = set()

        indexlist1=list([i+4,j+k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A1=list(board1[m[0],m[1]] for m in indexlist1)
        
        for a in range(len(A1) - window_size + 1):
            if A1[a:a+window_size] == target410:
                results.add(board19[i+4][j+4-(4-a)])
            if A1[a:a+window_size] == target411:
                results.add(board19[i+4][j+4+a])
            if A1[a:a+window_size] == target42:
                results.add(board19[i+4][j+4-(2-a)])
            if A1[a:a+window_size] == target430:
                results.add(board19[i+4][j+4-(3-a)])
            if A1[a:a+window_size] == target431:
                results.add(board19[i+4][j+4-(1-a)])
        
        indexlist2=list([i+k+4,j+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A2=list(board1[m[0],m[1]] for m in indexlist2)
        
        for a in range(len(A2) - window_size + 1):
            if A2[a:a+window_size] == target410:
                results.add(board19[i+4-(4-a)][j+4])
            if A2[a:a+window_size] == target411:
                results.add(board19[i+4+a][j+4])
            if A2[a:a+window_size] == target42:
                results.add(board19[i+4-(2-a)][j+4])
            if A2[a:a+window_size] == target430:
                results.add(board19[i+4-(3-a)][j+4])
            if A2[a:a+window_size] == target431:
                results.add(board19[i+4-(1-a)][j+4])
        
        indexlist3=list([i+k+4,j+k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A3=list(board1[m[0],m[1]] for m in indexlist3)

        for a in range(len(A3) - window_size + 1):
            if A3[a:a+window_size] == target410:
                results.add(board19[i+4-(4-a)][j+4-(4-a)])
            if A3[a:a+window_size] == target411:
                results.add(board19[i+4+a][j+4+a])
            if A3[a:a+window_size] == target42:
                results.add(board19[i+4-(2-a)][j+4-(2-a)])
            if A3[a:a+window_size] == target430:
                results.add(board19[i+4-(3-a)][j+4-(3-a)])
            if A3[a:a+window_size] == target431:
                results.add(board19[i+4-(1-a)][j+4-(1-a)])
        
        indexlist4=list([i+k+4,j-k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A4=list(board1[m[0],m[1]] for m in indexlist4)

        for a in range(len(A4) - window_size + 1):
            if A4[a:a+window_size] == target410:
                results.add(board19[i+4-(4-a)][j+4+(4-a)])
            if A4[a:a+window_size] == target411:
                results.add(board19[i+4+a][j+4-a])
            if A4[a:a+window_size] == target42:
                results.add(board19[i+4-(2-a)][j+4+(2-a)])
            if A4[a:a+window_size] == target430:
                results.add(board19[i+4-(3-a)][j+4+(3-a)])
            if A4[a:a+window_size] == target431:
                results.add(board19[i+4-(1-a)][j+4+(1-a)])
        
        return results

    def checkai4(self, board):
        """
        ���ai4
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # ֵ���� -1,0,1
    
        array11 = np.concatenate((array_2d[-4:,-4:], array_2d[-4:,:],array_2d[-4:,:4]), axis=1)
        array12 = np.concatenate((array_2d[:,-4:], array_2d,array_2d[:,:4]), axis=1)
        array13 = np.concatenate((array_2d[:4,-4:], array_2d[:4,:],array_2d[:4,:4]), axis=1)
    
        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19��19 �ж����� ֵ���� -1,0,1
        
        n=board.height
        
        tent=board.last_last_change["last_last"]
        i=board.move_to_location(tent)[0]
        j=board.move_to_location(tent)[1]
        
        player=array_2d[i][j]
        
        target410=[0, player, player, player, player]
        target411=[player, player, player, player, 0]
        target42=[player,player,0,player,player]
        target430=[player,0,player,player,player]
        target431=[player,player,player,0,player]
        
        window_size=len(target410)
        results = set()

        indexlist1=list([i+4,j+k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A1=list(board1[m[0],m[1]] for m in indexlist1)
        
        for a in range(len(A1) - window_size + 1):
            if A1[a:a+window_size] == target410:
                results.add(board19[i+4][j+4-(4-a)])
            if A1[a:a+window_size] == target411:
                results.add(board19[i+4][j+4+a])
            if A1[a:a+window_size] == target42:
                results.add(board19[i+4][j+4-(2-a)])
            if A1[a:a+window_size] == target430:
                results.add(board19[i+4][j+4-(3-a)])
            if A1[a:a+window_size] == target431:
                results.add(board19[i+4][j+4-(1-a)])
        
        indexlist2=list([i+k+4,j+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A2=list(board1[m[0],m[1]] for m in indexlist2)
        
        for a in range(len(A2) - window_size + 1):
            if A2[a:a+window_size] == target410:
                results.add(board19[i+4-(4-a)][j+4])
            if A2[a:a+window_size] == target411:
                results.add(board19[i+4+a][j+4])
            if A2[a:a+window_size] == target42:
                results.add(board19[i+4-(2-a)][j+4])
            if A2[a:a+window_size] == target430:
                results.add(board19[i+4-(3-a)][j+4])
            if A2[a:a+window_size] == target431:
                results.add(board19[i+4-(1-a)][j+4])
        
        indexlist3=list([i+k+4,j+k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A3=list(board1[m[0],m[1]] for m in indexlist3)

        for a in range(len(A3) - window_size + 1):
            if A3[a:a+window_size] == target410:
                results.add(board19[i+4-(4-a)][j+4-(4-a)])
            if A3[a:a+window_size] == target411:
                results.add(board19[i+4+a][j+4+a])
            if A3[a:a+window_size] == target42:
                results.add(board19[i+4-(2-a)][j+4-(2-a)])
            if A3[a:a+window_size] == target430:
                results.add(board19[i+4-(3-a)][j+4-(3-a)])
            if A3[a:a+window_size] == target431:
                results.add(board19[i+4-(1-a)][j+4-(1-a)])
        
        indexlist4=list([i+k+4,j-k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A4=list(board1[m[0],m[1]] for m in indexlist4)

        for a in range(len(A4) - window_size + 1):
            if A4[a:a+window_size] == target410:
                results.add(board19[i+4-(4-a)][j+4+(4-a)])
            if A4[a:a+window_size] == target411:
                results.add(board19[i+4+a][j+4-a])
            if A4[a:a+window_size] == target42:
                results.add(board19[i+4-(2-a)][j+4+(2-a)])
            if A4[a:a+window_size] == target430:
                results.add(board19[i+4-(3-a)][j+4+(3-a)])
            if A4[a:a+window_size] == target431:
                results.add(board19[i+4-(1-a)][j+4+(1-a)])
        
        return results

    def checkp3(self, board):
        """
        ������3
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # ֵ���� -1,0,1

        array11 = np.concatenate((array_2d[-4:,-4:], array_2d[-4:,:],array_2d[-4:,:4]), axis=1)
        array12 = np.concatenate((array_2d[:,-4:], array_2d,array_2d[:,:4]), axis=1)
        array13 = np.concatenate((array_2d[:4,-4:], array_2d[:4,:],array_2d[:4,:4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19��19 �ж����� ֵ���� -1,0,1

        n=board.height

        tent=board.last_change["last"]
        i=board.move_to_location(tent)[0]
        j=board.move_to_location(tent)[1]
        
        player=array_2d[i][j]
        
        target30=[0, player, player, player, 0]
        target31=[0, player,0, player,player,0]
        target32=[0, player,player,0, player,0]

        window_size1=len(target30)
        window_size2=len(target31)
        
        results = set()

        indexlist1=list([i+4,j+k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A1=list(board1[m[0],m[1]] for m in indexlist1)

        for a in range(len(A1) - window_size1 + 1):
            if A1[a:a+window_size1] == target30:
                results.add(board19[i+4][j+4-(4-a)])
                results.add(board19[i+4][j+4+a])
        
        for a in range(len(A1) - window_size2 + 1):
            if A1[a:a+window_size2] == target31:
                results.add(board19[i+4][j+4-(4-a)])
                results.add(board19[i+4][j+4-(2-a)])
                results.add(board19[i+4][j+4+(1+a)])
            
            if A1[a:a+window_size2] == target32:
                results.add(board19[i+4][j+4-(4-a)])
                results.add(board19[i+4][j+4-(1-a)])
                results.add(board19[i+4][j+4+(1+a)])
        
        indexlist2=list([i+k+4,j+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A2=list(board1[m[0],m[1]] for m in indexlist2)

        for a in range(len(A2) - window_size1 + 1):
            if A2[a:a+window_size1] == target30:
                results.add(board19[i+4-(4-a)][j+4])
                results.add(board19[i+4+a][j+4])
        
        for a in range(len(A2) - window_size2 + 1):
            if A2[a:a+window_size2] == target31:
                results.add(board19[i+4-(4-a)][j+4])
                results.add(board19[i+4-(2-a)][j+4])
                results.add(board19[i+4+(1+a)][j+4])
            
            if A2[a:a+window_size2] == target32:
                results.add(board19[i+4-(4-a)][j+4])
                results.add(board19[i+4-(1-a)][j+4])
                results.add(board19[i+4+(1+a)][j+4])
        
        indexlist3=list([i+k+4,j+k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A3=list(board1[m[0],m[1]] for m in indexlist3)

        for a in range(len(A3) - window_size1 + 1):
            if A3[a:a+window_size1] == target30:
                results.add(board19[i+4-(4-a)][j+4-(4-a)])
                results.add(board19[i+4+a][j+4+a])
        
        for a in range(len(A3) - window_size2 + 1):
            if A3[a:a+window_size2] == target31:
                results.add(board19[i+4-(4-a)][j+4-(4-a)])
                results.add(board19[i+4-(2-a)][j+4-(2-a)])
                results.add(board19[i+4+(1+a)][j+4+(1+a)])
            
            if A3[a:a+window_size2] == target32:
                results.add(board19[i+4-(4-a)][j+4-(4-a)])
                results.add(board19[i+4-(1-a)][j+4-(1-a)])
                results.add(board19[i+4+(1+a)][j+4+(1+a)])

        indexlist4=list([i+k+4,j-k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A4=list(board1[m[0],m[1]] for m in indexlist4)

        for a in range(len(A4) - window_size1 + 1):
            if A4[a:a+window_size1] == target30:
                results.add(board19[i+4-(4-a)][j+4+(4-a)])
                results.add(board19[i+4+a][j+4-a])
        
        for a in range(len(A4) - window_size2 + 1):
            if A4[a:a+window_size2] == target31:
                results.add(board19[i+4-(4-a)][j+4+(4-a)])
                results.add(board19[i+4-(2-a)][j+4+(2-a)])
                results.add(board19[i+4+(1+a)][j+4-(1+a)])
            
            if A4[a:a+window_size2] == target32:
                results.add(board19[i+4-(4-a)][j+4+(4-a)])
                results.add(board19[i+4-(1-a)][j+4+(1-a)])
                results.add(board19[i+4+(1+a)][j+4-(1+a)])

        return results

    def checkai3(self, board):
        """
        ������3
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # ֵ���� -1,0,1

        array11 = np.concatenate((array_2d[-4:,-4:], array_2d[-4:,:],array_2d[-4:,:4]), axis=1)
        array12 = np.concatenate((array_2d[:,-4:], array_2d,array_2d[:,:4]), axis=1)
        array13 = np.concatenate((array_2d[:4,-4:], array_2d[:4,:],array_2d[:4,:4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19��19 �ж����� ֵ���� -1,0,1

        n=board.height

        tent=board.last_last_change["last_last"]
        i=board.move_to_location(tent)[0]
        j=board.move_to_location(tent)[1]
        
        player=array_2d[i][j]
        
        target30=[0, player, player,player, 0]
        target31=[0, player,0, player,player,0]
        target32=[0, player,player,0, player,0]

        window_size1=len(target30)
        window_size2=len(target31)
        
        results = set()

        indexlist1=list([i+4,j+k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A1=list(board1[m[0],m[1]] for m in indexlist1)

        for a in range(len(A1) - window_size1 + 1):
            if A1[a:a+window_size1] == target30:
                results.add(board19[i+4][j+4-(4-a)])
                results.add(board19[i+4][j+4+a])
        
        for a in range(len(A1) - window_size2 + 1):
            if A1[a:a+window_size2] == target31:
                results.add(board19[i+4][j+4-(4-a)])
                results.add(board19[i+4][j+4-(2-a)])
                results.add(board19[i+4][j+4+(1+a)])
            
            if A1[a:a+window_size2] == target32:
                results.add(board19[i+4][j+4-(4-a)])
                results.add(board19[i+4][j+4-(1-a)])
                results.add(board19[i+4][j+4+(1+a)])
        
        indexlist2=list([i+k+4,j+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A2=list(board1[m[0],m[1]] for m in indexlist2)

        for a in range(len(A2) - window_size1 + 1):
            if A2[a:a+window_size1] == target30:
                results.add(board19[i+4-(4-a)][j+4])
                results.add(board19[i+4+a][j+4])
        
        for a in range(len(A2) - window_size2 + 1):
            if A2[a:a+window_size2] == target31:
                results.add(board19[i+4-(4-a)][j+4])
                results.add(board19[i+4-(2-a)][j+4])
                results.add(board19[i+4+(1+a)][j+4])
            
            if A2[a:a+window_size2] == target32:
                results.add(board19[i+4-(4-a)][j+4])
                results.add(board19[i+4-(1-a)][j+4])
                results.add(board19[i+4+(1+a)][j+4])
        
        indexlist3=list([i+k+4,j+k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A3=list(board1[m[0],m[1]] for m in indexlist3)

        for a in range(len(A3) - window_size1 + 1):
            if A3[a:a+window_size1] == target30:
                results.add(board19[i+4-(4-a)][j+4-(4-a)])
                results.add(board19[i+4+a][j+4+a])
        
        for a in range(len(A3) - window_size2 + 1):
            if A3[a:a+window_size2] == target31:
                results.add(board19[i+4-(4-a)][j+4-(4-a)])
                results.add(board19[i+4-(2-a)][j+4-(2-a)])
                results.add(board19[i+4+(1+a)][j+4+(1+a)])
            
            if A3[a:a+window_size2] == target32:
                results.add(board19[i+4-(4-a)][j+4-(4-a)])
                results.add(board19[i+4-(1-a)][j+4-(1-a)])
                results.add(board19[i+4+(1+a)][j+4+(1+a)])

        indexlist4=list([i+k+4,j-k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
        A4=list(board1[m[0],m[1]] for m in indexlist4)

        for a in range(len(A4) - window_size1 + 1):
            if A4[a:a+window_size1] == target30:
                results.add(board19[i+4-(4-a)][j+4+(4-a)])
                results.add(board19[i+4+a][j+4-a])
        
        for a in range(len(A4) - window_size2 + 1):
            if A4[a:a+window_size2] == target31:
                results.add(board19[i+4-(4-a)][j+4+(4-a)])
                results.add(board19[i+4-(2-a)][j+4+(2-a)])
                results.add(board19[i+4+(1+a)][j+4-(1+a)])
            
            if A4[a:a+window_size2] == target32:
                results.add(board19[i+4-(4-a)][j+4+(4-a)])
                results.add(board19[i+4-(1-a)][j+4+(1-a)])
                results.add(board19[i+4+(1+a)][j+4-(1+a)])

        return results

    def skipf(self, board):
        indic = self.checkai4(board)
        if len(indic) != 0:
            return list(indic)
        else:
            indic2 = self.checkp4(board)
            if len(indic2) != 0:
                return list(indic2)
            else:
                indic3 = self.checkp3(board)
                indic4 = self.checkai3(board)
                if len(indic3) != 0 and len(indic4) == 0:
                    return list(indic3)
                elif len(indic3) != 0 and len(indic4) != 0:
                    return list(indic3)+list(indic4)
                else:
                    fb = self.checkpforbid(board)
                    if len(fb) != 0 and len(indic4) != 0:
                        return list(fb)+list(indic4)
                    elif len(fb) != 0 and len(indic4) == 0:
                        return list(fb)
                    # elif len(fb) == 0 and len(indic4) != 0:
                    #     return list(indic4)
                    else:
                        return False

    def adjacent2(self, board):
      
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        adjacents = set()
        width = board.width
        height = board.height
     
        for m in moved:
            h = m // width
            w = m % width
            if w < width - 1:
                adjacents.add(m + 1) # ��
            if w == width - 1:
                adjacents.add(m + 1 - width) # �ҵ���  
            if w > 0:
                adjacents.add(m - 1) # ��
            if w == 0:
                adjacents.add(m - 1 + width) # ����
            if h < height - 1:
                adjacents.add(m + width) # ��
            if h == height - 1:
                adjacents.add(m + width - height*width) # �µ���
            if h > 0:
                adjacents.add(m - width) # ��
            if h == 0:
                adjacents.add(m - width + height*width) # �ϵ���
            if w < width - 1 and h < height - 1:
                adjacents.add(m + width + 1) # ����
            if w == width - 1 and h == height - 1:
                adjacents.add(m + width + 1 - width - height*width) # ���µ�����
            if w > 0 and h < height - 1:
                adjacents.add(m + width - 1) # ����
            if w == 0 and h == height - 1:
                adjacents.add(m + width - 1 + width - height*width) # ���µ�����
            if w < width - 1 and h > 0:
                adjacents.add(m - width + 1) # ����
            if w == width - 1 and h == 0:
                adjacents.add(m - width + 1 - width + height*width) # ���ϵ�����
            if w > 0 and h > 0:
                adjacents.add(m - width - 1) # ����
            if w == 0 and h == 0:
                adjacents.add(m - width - 1 + width + height*width) # ���ϵ�����
     
        adjacents = list(set(adjacents) - set(moved))

        return adjacents

    def adjacent3(self, board):
      
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        moved += self.adjacent2(board)
        adjacents = set()
        width = board.width
        height = board.height
     
        for m in moved:
            h = m // width
            w = m % width
            if w < width - 1:
                adjacents.add(m + 1) # ��
            if w == width - 1:
                adjacents.add(m + 1 - width) # �ҵ���  
            if w > 0:
                adjacents.add(m - 1) # ��
            if w == 0:
                adjacents.add(m - 1 + width) # ����
            if h < height - 1:
                adjacents.add(m + width) # ��
            if h == height - 1:
                adjacents.add(m + width - height*width) # �µ���
            if h > 0:
                adjacents.add(m - width) # ��
            if h == 0:
                adjacents.add(m - width + height*width) # �ϵ���
            if w < width - 1 and h < height - 1:
                adjacents.add(m + width + 1) # ����
            if w == width - 1 and h == height - 1:
                adjacents.add(m + width + 1 - width - height*width) # ���µ�����
            if w > 0 and h < height - 1:
                adjacents.add(m + width - 1) # ����
            if w == 0 and h == height - 1:
                adjacents.add(m + width - 1 + width - height*width) # ���µ�����
            if w < width - 1 and h > 0:
                adjacents.add(m - width + 1) # ����
            if w == width - 1 and h == 0:
                adjacents.add(m - width + 1 - width + height*width) # ���ϵ�����
            if w > 0 and h > 0:
                adjacents.add(m - width - 1) # ����
            if w == 0 and h == 0:
                adjacents.add(m - width - 1 + width + height*width) # ���ϵ�����
     
        adjacents = list(set(adjacents) - set(moved))

        return adjacents

    def checkpforbid(self, board):
        pool = self.adjacent2(board)+self.adjacent3(board)
        forbidmove = set()
        for i in pool:
            board_copy = copy.deepcopy(board)
            board_copy.update(-self.player, i)
            g3 = self.checkp3(board_copy)
            g4 = self.checkp4(board_copy)
            if len(g3) > 3 or len(g4) > 2 or (len(g4) > 0 and len(g3) > 0):
                forbidmove.add(i)
        
        return forbidmove
        
    
class skcBoard(object):
    """
    board for game
    """

    def __init__(self, width=11, height=11, n_in_row=5):
        self.width = width
        self.height = height
        self.states = {}  # ��¼��ǰ���̵�״̬������λ�ã�ֵ�����ӣ��������������ʾ��������
        self.last_change = {"last": -1}
        self.last_last_change = {"last_last": -1}
        self.n_in_row = n_in_row  # ��ʾ������ͬ����������һ������ʤ��
        self.steps = 0

    def init_board(self):
        self.availables = list(range(self.width * self.height))  # ��ʾ���������кϷ���λ�ã�����򵥵���Ϊ�յ�λ�ü��Ϸ�

        for m in self.availables:
            self.states[m] = 0  # 0��ʾ��ǰλ��Ϊ��

    def move_to_location(self, move):
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if (len(location) != 2):
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if (move not in range(self.width * self.height)):
            return -1
        return move

    def update(self, player, move):  # player��move�����ӣ���������
        self.states[move] = player
        self.availables.remove(move)
        self.last_last_change["last_last"] = self.last_change["last"]
        self.last_change["last"] = move
        self.steps += 1

class TreeNode(object):
    def __init__(self, parent=None, pre_pos=None, board=None, player=None):
        self.pre_pos = pre_pos  # [0-121]    # ���������̵Ľ���±�

        self.parent = parent  # �����
        self.children = list()  # �ӽ��
        # self.player = player  # ���ⲽ������

        self.not_visit_pos = None  # δ���ʹ��Ľڵ�

        self.board = board  # ÿ������Ӧһ������״̬

        self.num_of_visit = 0  # ���ʴ���N
        # self.num_of_win = 0                     # ʤ������M ��Ҫʵʱ����
        self.num_of_wins = defaultdict(int)  # ��¼�ý��ģ��İ��ӡ����ӵ�ʤ������(defaultdict: ���ֵ����key�����ڵ�������ʱ������0)

    def fully_expanded(self):
        """
        :return: True: �ý���Ѿ���ȫ��չ, False: �ý��δ��ȫ��չ
        """
        if self.not_visit_pos is None:  # ���δ���ʹ��Ľ��ΪNone(��ʼ��ΪNone)��δ������չ��
            self.not_visit_pos = set(self.board.availables) #- set([ele.pre_pos for ele in self.children]) # �õ�����Ϊ�ý����չ���������±�


        return True if (len(self.not_visit_pos) == 0 and len(self.children) != 0) else False

    def pick_univisted(self, player):
        """ѡ��һ��δ���ʵĽ��"""
        # print(len(self.not_visit_pos))
        random_index = np.random.randint(0, len(self.not_visit_pos))  # ���ѡ��һ��δ���ʵĽ�㣨random.randint: �����䣩

        move_pos = list(self.not_visit_pos)[random_index]  # �õ�һ�������δ���ʽ��, �������е�δ���ʽ����ɾ��
        self.not_visit_pos.remove(move_pos)

        # self.board.update(player, move_pos)
        new_board = copy.deepcopy(self.board)  # ģ�����Ӳ�����������
        new_board.update(player, move_pos)

        new_node = TreeNode(parent=self, pre_pos=move_pos, board=new_board)  # �����̰��½��
        self.children.append(new_node)
        return new_node

    def pick_random(self,player):
        """ѡ����ĺ��ӽ�����չ"""
        possible_moves = self.board.availables  # �������ӵĵ�λ

        random_index = np.random.randint(0, len(possible_moves) - 1)  # ���ѡ��һ���������ӵĵ�λ��random.randint: ������test��
        move_pos = possible_moves[random_index]  # �õ�һ������Ŀ������ӵĵ�λ

        # self.board.update(player, move_pos)
        new_board = copy.deepcopy(self.board)  # ģ�����Ӳ�����������
        new_board.update(player, move_pos)


        new_node = TreeNode(parent=self, pre_pos=move_pos, board=new_board)  # �����̰��½��
        return new_node


    def num_of_win(self, playturn):
        # print('playturn:',playturn)
        wins = self.num_of_wins[playturn]  # ���ӽ�������״̬���ڸ��ڵ��next_player֮���γ�
        # loses = self.num_of_wins[-playturn]
        return wins

    def best_uct(self,playturn, c_param=1.98):
        """����һ���Լ���õĺ��ӽ�㣨����UCT���бȽϣ�"""
        uct_of_children = np.array(list([
            (child.num_of_win(playturn) / child.num_of_visit) + c_param * np.sqrt(
                np.log(self.num_of_visit) / child.num_of_visit)
            for child in self.children
        ]))
        best_index = np.argmax(uct_of_children)
        # max_uct = max(uct_of_children)
        # best_index = np.where(uct_of_children == max_uct)     # ��ȡ���uct���±�
        # best_index = np.random.choice(best_index[0])        # ���ѡȡһ��ӵ�����uct�ĺ���
        return self.children[best_index]

    def __str__(self):
        return "pre_pos: {}\t pre_player: {}\t num_of_visit: {}\t num_of_wins: {}" \
            .format(self.pre_pos, self.board.last_change,
                    self.num_of_visit, dict(self.num_of_wins))

class skcMCTS(object):
    """
    AI player, use Monte Carlo Tree Search with UCB
    """

    def __init__(self, board, play_turn, n_in_row=5, time=5, max_actions=1000):

        self.board = board
        self.play_turn = play_turn  # ����˳��
        self.calculation_time = float(time)  # �������ʱ��
        self.max_actions = max_actions  # ÿ��ģ��Ծ������еĲ���
        self.n_in_row = n_in_row

        self.player = play_turn[0]  # �ֵ����Գ��֣����Գ���˳���е�һ�����ǵ���
        self.confident = 1.96  # UCB�еĳ��� 1.96

        self.max_depth = 1
        self.nodenum = 0
        self.skip = False

    def fully_expanded(self, node):
        """
        :return: True: �ý���Ѿ���ȫ��չ, False: �ý��δ��ȫ��չ
        """
        if node.not_visit_pos is None:
            # node.not_visit_pos = set(node.board.availables)
            node.not_visit_pos = self.adjacent2(node.board)+self.adjacent3(node.board)
        return True if (len(node.not_visit_pos) == 0 and len(node.children) != 0) else False

    def traverse(self, node, play_turn):
        """
        ��α����ý�㼰���ӽ�㣬����Ҷ�ӽ�㣬����δ��ȫ��չ�Ľ������������չ
        """
        player = self.get_player(play_turn)  # ��ȡ��ǰ���ֵ����
        while node.fully_expanded():  # �ý���Ѿ���ȫ��չ, ѡ��һ��UCT��ߵĺ���
            node = node.best_uct(player)
            player = self.get_player(play_turn)
        # ����δ�����չ�Ľ����˳�ѭ�����ȼ���Ƿ�ΪҶ�ӽ��
        if self.check_winner(node.board) != 0:  # ��Ҷ�ӽ��(node is terminal)
            return node
        else:  # ����Ҷ�ӽ���һ�û�к���(in case no children are present)
            # self.nodenum += 1
            return self.expand_policy(node, player)
            # return node.pick_univisted(player)

    def expand_policy(self, node, player):
        plays = [ele.pre_pos for ele in node.children]
        # plays = set(node.board.availables) - set(node.not_visit_pos)
        adjacents = self.adjacent_moves(node.board, player, plays)
        # adjacents = list(set(adjacents) - set(plays))
        if len(adjacents):
            move = random.choice(adjacents)
        else:
            peripherals = []
            for move in node.board.availables:
                if move not in plays:
                    peripherals.append(move)  # û��ͳ����Ϣ����Χλ��
            move = random.choice(peripherals)
        # print('plays:',plays,'move:',move, 'adj:',adjacents)

        node.not_visit_pos.remove(move)
        new_board = copy.deepcopy(node.board)  # ģ�����Ӳ�����������
        new_board.update(player, move)

        new_node = TreeNode(parent=node, pre_pos=move, board=new_board)
        node.children.append(new_node)
        return new_node

    # def rollout(self, node, play_turn,depth=0):
    #     player = self.get_player(play_turn)  # ��ȡ��ǰ���ֵ����
    #     while True:
    #         depth += 1
    #         game_result = self.check_winner(node.board)
    #         # print('depth:',depth,'game_result:',game_result,'player:',player)
    #         if game_result == 0:  # ����Ҷ�ӽ��, ����ģ��
    #             node = self.rollout_policy(node, player)
    #         else:  # ��Ҷ�ӽ�㣬����
    #             break
    #         player = self.get_player(play_turn)
    #     self.max_depth = max(depth, self.max_depth)
    #     return self.check_winner(node.board)
    def rollout(self, node, play_turn,depth=0):
        player = self.get_player(play_turn)  # ��ȡ��ǰ���ֵ����
        board = copy.deepcopy(node.board)
        while True:
            depth += 1
            game_result = self.check_winner(board)
            # print('depth:',depth,'game_result:',game_result,'player:',player)
            if game_result == 0:  # ����Ҷ�ӽ��, ����ģ��
                board = self.rollout_policy(board, player)
            else:  # ��Ҷ�ӽ�㣬����
                break
            player = self.get_player(play_turn)
        self.max_depth = max(depth, self.max_depth)
        return self.check_winner(board)
    def rollout_policy(self, board, player):
        plays = []
        adjacents = self.adjacent_moves(board, player, plays)
        if len(adjacents):
            move = random.choice(adjacents)
        else:
            peripherals = []
            for move in board.availables:
                if move not in plays:
                    peripherals.append(move)  # û��ͳ����Ϣ����Χλ��
            move = random.choice(peripherals)

        board.update(player, move)

        return board

    # def rollout_policy(self, node, player):
    #     plays = []
    #     adjacents = self.adjacent_moves(node.board, player, plays)
    #     if len(adjacents):
    #         move = random.choice(adjacents)
    #     else:
    #         peripherals = []
    #         for move in node.board.availables:
    #             if move not in plays:
    #                 peripherals.append(move)  # û��ͳ����Ϣ����Χλ��
    #         move = random.choice(peripherals)
    #
    #     new_board = copy.deepcopy(node.board)  # ģ�����Ӳ�����������
    #     new_board.update(player, move)
    #
    #     new_node = TreeNode(parent=node, pre_pos=move, board=new_board)
    #     return new_node

    def backpropagate(self, node, result):
        node.num_of_visit += 1
        node.num_of_wins[result] += 1
        if node.parent:  # ������Ǹ���㣬����������丸�ڵ�
            self.backpropagate(node.parent, result)

    def best_child(self, node):
        visit_num_of_children = np.array(list([child.num_of_visit for child in node.children]))
        best_index = np.argmax(visit_num_of_children)  # ��ȡ���uct���±�
        node = node.children[best_index]
        # print('root_child_node_info: ', node.num_of_visit, node.num_of_wins)
        return node
    def monte_carlo_tree_search(self, board_ori, pre_pos, simulations):
        board = copy.deepcopy(board_ori)
        root = TreeNode(board=board, pre_pos=pre_pos)  # ����㣬������޸���
        # for i in range(700):  # �൱��(while resources_left(time, computational power):)����Դ����
        begin = time.time()
        while time.time() - begin < self.calculation_time:
            self.max_depth = 1
            play_turn_copy = copy.deepcopy(self.play_turn)  # ÿ��ģ�ⶼ���밴�չ̶���˳����У����Խ��������ֹ˳���޸�
            leaf = self.traverse(root, play_turn_copy)  # ѡ�����չ��leaf = unvisited node����������㣩
            simulation_result = self.rollout(leaf, play_turn_copy)  # ģ��
            self.backpropagate(leaf, simulation_result)  # ���򴫲�
            simulations += 1
        print('simulations/node number:',simulations)
        return root
        # return self.best_child(root).pre_pos
    def get_action(self):  # return move

        if len(self.board.availables) == 1:
            return self.board.availables[0]  # ����ֻʣ���һ������λ�ã�ֱ�ӷ���

        # self.nodenum = 0
        self.max_depth = 1
        self.skip = False
        simulations = 0
        root = self.monte_carlo_tree_search(self.board, self.board.last_change["last"], simulations)

        self.skip = self.skipf(self.board)
        # print('skip:',self.skip)
        # print('board:',self.board.availables)
        move, percent_wins = self.select_one_move(self.board,root)  # ѡ������ŷ�
        #
        location = self.board.move_to_location(move)
        print('Maximum depth searched:', self.max_depth)
        # print('Maximum node number:', self.nodenum)

        print("AI move: %d,%d\n" % (location[0], location[1]))
        print('AI move percent_wins: %f\n' % (percent_wins))

        return move
    def get_player(self, players):
        p = players.pop(0)
        players.append(p)
        return p

    def select_one_move(self, board, node):
        if self.skip and board.steps >3:
            limited = np.array(self.skip)
            visit_num_of_children = np.array(list([child.num_of_visit for child in node.children]))
            pre_pos_of_children = np.array(list([child.pre_pos for child in node.children]))
            idx = np.where(np.isin(pre_pos_of_children, limited))

            visit_num_of_children = visit_num_of_children[idx]
            win_num_of_children = np.array(list([child.num_of_win(self.play_turn[0]) for child in node.children]))[idx]
            percent_wins = win_num_of_children / visit_num_of_children
            ucbs = percent_wins + self.confident * np.sqrt(np.log(node.num_of_visit) / visit_num_of_children)

            best_index = np.argmax(ucbs)  # ��ȡ���ucb���±�
            node = np.array(node.children)[idx][best_index]
        else:
            if len(board.availables) >= 119:
                limited = np.array(self.adjacent2(board))
            else:
                limited = np.array(self.adjacent2(board)+self.adjacent3(board))

            visit_num_of_children = np.array(list([child.num_of_visit for child in node.children]))
            pre_pos_of_children = np.array(list([child.pre_pos for child in node.children]))

            # print('prepos:',pre_pos_of_children)
            # print('isin:',np.isin(pre_pos_of_children, limited))

            idx = np.where(np.isin(pre_pos_of_children, limited))
            visit_num_of_children = visit_num_of_children[idx]

            win_num_of_children = np.array(list([child.num_of_win(self.play_turn[0]) for child in node.children]))[idx]
            percent_wins = win_num_of_children / visit_num_of_children
            ucbs = percent_wins + self.confident * np.sqrt(np.log(node.num_of_visit) / visit_num_of_children)

            best_index = np.argmax(ucbs)  # ��ȡ���ucb���±�
            # print('idx:',idx,best_index)
            # print('node:',len(np.array(node.children)))
            node = np.array(node.children)[idx][best_index]

        print('win:',node.num_of_win(self.play_turn[0]),'visit:',node.num_of_visit, 'ucb:',ucbs)
        return node.pre_pos, node.num_of_win(self.play_turn[0]) / node.num_of_visit

    def check_winner(self, board):
        """
        ����Ƿ�����һ�ʤ
        """
        # print('board:',board.states[0])
        # print(np.array([board.states[key][0] for key in range(121)])[:10])
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)
        # print(array_2d)

        array11 = np.concatenate((array_2d[-4:, -4:], array_2d[-4:, :], array_2d[-4:, :4]), axis=1)
        array12 = np.concatenate((array_2d[:, -4:], array_2d, array_2d[:, :4]), axis=1)
        array13 = np.concatenate((array_2d[:4, -4:], array_2d[:4, :], array_2d[:4, :4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19��19 �ж�����

        n = board.height

        tent = board.last_change["last"]
        i = board.move_to_location(tent)[0]
        j = board.move_to_location(tent)[1]

        indexlist1 = list([i + 4, j + k + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A1 = list(board1[m[0], m[1]] for m in indexlist1)
        count = 0
        for num in A1:
            if num == array_2d[i][j]:
                count += 1
                if count == 5:
                    return array_2d[i][j]
            else:
                count = 0

        indexlist2 = list([i + k + 4, j + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A2 = list(board1[m[0], m[1]] for m in indexlist2)
        count = 0
        for num in A2:
            if num == array_2d[i][j]:
                count += 1
                if count == 5:
                    return array_2d[i][j]
            else:
                count = 0

        indexlist3 = list([i + k + 4, j + k + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A3 = list(board1[m[0], m[1]] for m in indexlist3)
        count = 0
        for num in A3:
            if num == array_2d[i][j]:
                count += 1
                if count == 5:
                    return array_2d[i][j]
            else:
                count = 0

        indexlist4 = list([i - k + 4, j + k + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A4 = list(board1[m[0], m[1]] for m in indexlist4)
        count = 0
        for num in A4:
            if num == array_2d[i][j]:
                count += 1
                if count == 5:
                    return array_2d[i][j]
            else:
                count = 0

        for a in range(n):
            for b in range(n):
                if array_2d[a][b] == 0:
                    return 0
        return 2



    def adjacent_moves(self, board, player, plays):
        """
        ��ȡ��ǰ������������ӵ��ڽ�λ����û��ͳ����Ϣ��λ��
        """
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        adjacents = set()
        width = board.width
        height = board.height

        # for m in moved:
        #     h = m // width + 4
        #     w = m % width + 4
        #     periphery = {board19[h+1][w], board19[h-1][w], board19[h][w+1], board19[h][w-1],
        #                  board19[h-1][w-1], board19[h-1][w+1], board19[h+1][w-1], board19[h+1][w+1]}
        #     adjacents.update(periphery)


        for m in moved:
            h = m // width
            w = m % width
            if w < width - 1:
                adjacents.add(m + 1)  # ��
            if w == width - 1:
                adjacents.add(m + 1 - width)  # �ҵ���

            # if w <= width - 1:
            #     adjacents.add(m + 1 - width * (w//width)) # ��/�ҵ���

            if w > 0:
                adjacents.add(m - 1)  # ��
            if w == 0:
                adjacents.add(m - 1 + width)  # ����
            if h < height - 1:
                adjacents.add(m + width)  # ��
            if h == height - 1:
                adjacents.add(m + width - height * width)  # �µ���
            if h > 0:
                adjacents.add(m - width)  # ��
            if h == 0:
                adjacents.add(m - width + height * width)  # �ϵ���
            if w < width - 1 and h < height - 1:
                adjacents.add(m + width + 1)  # ����
            if w == width - 1 and h == height - 1:
                adjacents.add(m + width + 1 - width - height * width)  # ���µ�����
            if w > 0 and h < height - 1:
                adjacents.add(m + width - 1)  # ����
            if w == 0 and h == height - 1:
                adjacents.add(m + width - 1 + width - height * width)  # ���µ�����
            if w < width - 1 and h > 0:
                adjacents.add(m - width + 1)  # ����
            if w == width - 1 and h == 0:
                adjacents.add(m - width + 1 - width + height * width)  # ���ϵ�����
            if w > 0 and h > 0:
                adjacents.add(m - width - 1)  # ����
            if w == 0 and h == 0:
                adjacents.add(m - width - 1 + width + height * width)  # ���ϵ�����

        adjacents = list(set(adjacents) - set(moved))
        adjacents = list(set(adjacents) - set(plays))
        # for move in adjacents:
        #     if move in plays:
        #         adjacents.remove(move)
        return adjacents

    def checkp4(self, board):
        """
        ������4
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # ֵ���� -1,0,1

        array11 = np.concatenate((array_2d[-4:, -4:], array_2d[-4:, :], array_2d[-4:, :4]), axis=1)
        array12 = np.concatenate((array_2d[:, -4:], array_2d, array_2d[:, :4]), axis=1)
        array13 = np.concatenate((array_2d[:4, -4:], array_2d[:4, :], array_2d[:4, :4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19��19 �ж����� ֵ���� -1,0,1

        n = board.height

        tent = board.last_change["last"]
        if tent == -1:
            return []

        i = board.move_to_location(tent)[0]
        j = board.move_to_location(tent)[1]

        player = array_2d[i][j]

        target410 = [0, player, player, player, player]
        target411 = [player, player, player, player, 0]
        target42 = [player, player, 0, player, player]
        target430 = [player, 0, player, player, player]
        target431 = [player, player, player, 0, player]

        window_size = len(target410)
        results = set()

        indexlist1 = list([i + 4, j + k + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A1 = list(board1[m[0], m[1]] for m in indexlist1)

        for a in range(len(A1) - window_size + 1):
            if A1[a:a + window_size] == target410:
                results.add(board19[i + 4][j + 4 - (4 - a)])
            if A1[a:a + window_size] == target411:
                results.add(board19[i + 4][j + 4 + a])
            if A1[a:a + window_size] == target42:
                results.add(board19[i + 4][j + 4 - (2 - a)])
            if A1[a:a + window_size] == target430:
                results.add(board19[i + 4][j + 4 - (3 - a)])
            if A1[a:a + window_size] == target431:
                results.add(board19[i + 4][j + 4 - (1 - a)])

        indexlist2 = list([i + k + 4, j + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A2 = list(board1[m[0], m[1]] for m in indexlist2)

        for a in range(len(A2) - window_size + 1):
            if A2[a:a + window_size] == target410:
                results.add(board19[i + 4 - (4 - a)][j + 4])
            if A2[a:a + window_size] == target411:
                results.add(board19[i + 4 + a][j + 4])
            if A2[a:a + window_size] == target42:
                results.add(board19[i + 4 - (2 - a)][j + 4])
            if A2[a:a + window_size] == target430:
                results.add(board19[i + 4 - (3 - a)][j + 4])
            if A2[a:a + window_size] == target431:
                results.add(board19[i + 4 - (1 - a)][j + 4])

        indexlist3 = list([i + k + 4, j + k + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A3 = list(board1[m[0], m[1]] for m in indexlist3)

        for a in range(len(A3) - window_size + 1):
            if A3[a:a + window_size] == target410:
                results.add(board19[i + 4 - (4 - a)][j + 4 - (4 - a)])
            if A3[a:a + window_size] == target411:
                results.add(board19[i + 4 + a][j + 4 + a])
            if A3[a:a + window_size] == target42:
                results.add(board19[i + 4 - (2 - a)][j + 4 - (2 - a)])
            if A3[a:a + window_size] == target430:
                results.add(board19[i + 4 - (3 - a)][j + 4 - (3 - a)])
            if A3[a:a + window_size] == target431:
                results.add(board19[i + 4 - (1 - a)][j + 4 - (1 - a)])

        indexlist4 = list([i + k + 4, j - k + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A4 = list(board1[m[0], m[1]] for m in indexlist4)

        for a in range(len(A4) - window_size + 1):
            if A4[a:a + window_size] == target410:
                results.add(board19[i + 4 - (4 - a)][j + 4 + (4 - a)])
            if A4[a:a + window_size] == target411:
                results.add(board19[i + 4 + a][j + 4 - a])
            if A4[a:a + window_size] == target42:
                results.add(board19[i + 4 - (2 - a)][j + 4 + (2 - a)])
            if A4[a:a + window_size] == target430:
                results.add(board19[i + 4 - (3 - a)][j + 4 + (3 - a)])
            if A4[a:a + window_size] == target431:
                results.add(board19[i + 4 - (1 - a)][j + 4 + (1 - a)])

        return results

    def checkai4(self, board):
        """
        ���ai4
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # ֵ���� -1,0,1

        array11 = np.concatenate((array_2d[-4:, -4:], array_2d[-4:, :], array_2d[-4:, :4]), axis=1)
        array12 = np.concatenate((array_2d[:, -4:], array_2d, array_2d[:, :4]), axis=1)
        array13 = np.concatenate((array_2d[:4, -4:], array_2d[:4, :], array_2d[:4, :4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19��19 �ж����� ֵ���� -1,0,1

        n = board.height

        tent = board.last_last_change["last_last"]
        if tent == -1:
            return []

        i = board.move_to_location(tent)[0]
        j = board.move_to_location(tent)[1]

        # print('tent:', tent, 'i:', i, 'j:', j)

        player = array_2d[i][j]

        target410 = [0, player, player, player, player]
        target411 = [player, player, player, player, 0]
        target42 = [player, player, 0, player, player]
        target430 = [player, 0, player, player, player]
        target431 = [player, player, player, 0, player]

        window_size = len(target410)
        results = set()

        indexlist1 = list([i + 4, j + k + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A1 = list(board1[m[0], m[1]] for m in indexlist1)

        for a in range(len(A1) - window_size + 1):
            if A1[a:a + window_size] == target410:
                results.add(board19[i + 4][j + 4 - (4 - a)])
            if A1[a:a + window_size] == target411:
                results.add(board19[i + 4][j + 4 + a])
            if A1[a:a + window_size] == target42:
                results.add(board19[i + 4][j + 4 - (2 - a)])
            if A1[a:a + window_size] == target430:
                results.add(board19[i + 4][j + 4 - (3 - a)])
            if A1[a:a + window_size] == target431:
                results.add(board19[i + 4][j + 4 - (1 - a)])

        indexlist2 = list([i + k + 4, j + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A2 = list(board1[m[0], m[1]] for m in indexlist2)

        for a in range(len(A2) - window_size + 1):
            if A2[a:a + window_size] == target410:
                results.add(board19[i + 4 - (4 - a)][j + 4])
            if A2[a:a + window_size] == target411:
                results.add(board19[i + 4 + a][j + 4])
            if A2[a:a + window_size] == target42:
                results.add(board19[i + 4 - (2 - a)][j + 4])
            if A2[a:a + window_size] == target430:
                results.add(board19[i + 4 - (3 - a)][j + 4])
            if A2[a:a + window_size] == target431:
                results.add(board19[i + 4 - (1 - a)][j + 4])

        indexlist3 = list([i + k + 4, j + k + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A3 = list(board1[m[0], m[1]] for m in indexlist3)

        for a in range(len(A3) - window_size + 1):
            if A3[a:a + window_size] == target410:
                results.add(board19[i + 4 - (4 - a)][j + 4 - (4 - a)])
            if A3[a:a + window_size] == target411:
                results.add(board19[i + 4 + a][j + 4 + a])
            if A3[a:a + window_size] == target42:
                results.add(board19[i + 4 - (2 - a)][j + 4 - (2 - a)])
            if A3[a:a + window_size] == target430:
                results.add(board19[i + 4 - (3 - a)][j + 4 - (3 - a)])
            if A3[a:a + window_size] == target431:
                results.add(board19[i + 4 - (1 - a)][j + 4 - (1 - a)])

        indexlist4 = list([i + k + 4, j - k + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A4 = list(board1[m[0], m[1]] for m in indexlist4)

        for a in range(len(A4) - window_size + 1):
            if A4[a:a + window_size] == target410:
                results.add(board19[i + 4 - (4 - a)][j + 4 + (4 - a)])
            if A4[a:a + window_size] == target411:
                results.add(board19[i + 4 + a][j + 4 - a])
            if A4[a:a + window_size] == target42:
                results.add(board19[i + 4 - (2 - a)][j + 4 + (2 - a)])
            if A4[a:a + window_size] == target430:
                results.add(board19[i + 4 - (3 - a)][j + 4 + (3 - a)])
            if A4[a:a + window_size] == target431:
                results.add(board19[i + 4 - (1 - a)][j + 4 + (1 - a)])

        return results

    def checkp3(self, board):
        """
        ������3
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # ֵ���� -1,0,1

        array11 = np.concatenate((array_2d[-4:, -4:], array_2d[-4:, :], array_2d[-4:, :4]), axis=1)
        array12 = np.concatenate((array_2d[:, -4:], array_2d, array_2d[:, :4]), axis=1)
        array13 = np.concatenate((array_2d[:4, -4:], array_2d[:4, :], array_2d[:4, :4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19��19 �ж����� ֵ���� -1,0,1

        n = board.height

        tent = board.last_change["last"]
        if tent == -1:
            return []
        i = board.move_to_location(tent)[0]
        j = board.move_to_location(tent)[1]

        player = array_2d[i][j]

        target30 = [0, player, player, player, 0]
        target31 = [0, player, 0, player, player, 0]
        target32 = [0, player, player, 0, player, 0]

        window_size1 = len(target30)
        window_size2 = len(target31)

        results = set()

        indexlist1 = list([i + 4, j + k + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A1 = list(board1[m[0], m[1]] for m in indexlist1)

        for a in range(len(A1) - window_size1 + 1):
            if A1[a:a + window_size1] == target30:
                results.add(board19[i + 4][j + 4 - (4 - a)])
                results.add(board19[i + 4][j + 4 + a])

        for a in range(len(A1) - window_size2 + 1):
            if A1[a:a + window_size2] == target31:
                results.add(board19[i + 4][j + 4 - (4 - a)])
                results.add(board19[i + 4][j + 4 - (2 - a)])
                results.add(board19[i + 4][j + 4 + (1 + a)])

            if A1[a:a + window_size2] == target32:
                results.add(board19[i + 4][j + 4 - (4 - a)])
                results.add(board19[i + 4][j + 4 - (1 - a)])
                results.add(board19[i + 4][j + 4 + (1 + a)])

        indexlist2 = list([i + k + 4, j + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A2 = list(board1[m[0], m[1]] for m in indexlist2)

        for a in range(len(A2) - window_size1 + 1):
            if A2[a:a + window_size1] == target30:
                results.add(board19[i + 4 - (4 - a)][j + 4])
                results.add(board19[i + 4 + a][j + 4])

        for a in range(len(A2) - window_size2 + 1):
            if A2[a:a + window_size2] == target31:
                results.add(board19[i + 4 - (4 - a)][j + 4])
                results.add(board19[i + 4 - (2 - a)][j + 4])
                results.add(board19[i + 4 + (1 + a)][j + 4])

            if A2[a:a + window_size2] == target32:
                results.add(board19[i + 4 - (4 - a)][j + 4])
                results.add(board19[i + 4 - (1 - a)][j + 4])
                results.add(board19[i + 4 + (1 + a)][j + 4])

        indexlist3 = list([i + k + 4, j + k + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A3 = list(board1[m[0], m[1]] for m in indexlist3)

        for a in range(len(A3) - window_size1 + 1):
            if A3[a:a + window_size1] == target30:
                results.add(board19[i + 4 - (4 - a)][j + 4 - (4 - a)])
                results.add(board19[i + 4 + a][j + 4 + a])

        for a in range(len(A3) - window_size2 + 1):
            if A3[a:a + window_size2] == target31:
                results.add(board19[i + 4 - (4 - a)][j + 4 - (4 - a)])
                results.add(board19[i + 4 - (2 - a)][j + 4 - (2 - a)])
                results.add(board19[i + 4 + (1 + a)][j + 4 + (1 + a)])

            if A3[a:a + window_size2] == target32:
                results.add(board19[i + 4 - (4 - a)][j + 4 - (4 - a)])
                results.add(board19[i + 4 - (1 - a)][j + 4 - (1 - a)])
                results.add(board19[i + 4 + (1 + a)][j + 4 + (1 + a)])

        indexlist4 = list([i + k + 4, j - k + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A4 = list(board1[m[0], m[1]] for m in indexlist4)

        for a in range(len(A4) - window_size1 + 1):
            if A4[a:a + window_size1] == target30:
                results.add(board19[i + 4 - (4 - a)][j + 4 + (4 - a)])
                results.add(board19[i + 4 + a][j + 4 - a])

        for a in range(len(A4) - window_size2 + 1):
            if A4[a:a + window_size2] == target31:
                results.add(board19[i + 4 - (4 - a)][j + 4 + (4 - a)])
                results.add(board19[i + 4 - (2 - a)][j + 4 + (2 - a)])
                results.add(board19[i + 4 + (1 + a)][j + 4 - (1 + a)])

            if A4[a:a + window_size2] == target32:
                results.add(board19[i + 4 - (4 - a)][j + 4 + (4 - a)])
                results.add(board19[i + 4 - (1 - a)][j + 4 + (1 - a)])
                results.add(board19[i + 4 + (1 + a)][j + 4 - (1 + a)])

        return results

    def checkai3(self, board):
        """
        ������3
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # ֵ���� -1,0,1

        array11 = np.concatenate((array_2d[-4:, -4:], array_2d[-4:, :], array_2d[-4:, :4]), axis=1)
        array12 = np.concatenate((array_2d[:, -4:], array_2d, array_2d[:, :4]), axis=1)
        array13 = np.concatenate((array_2d[:4, -4:], array_2d[:4, :], array_2d[:4, :4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19��19 �ж����� ֵ���� -1,0,1

        n = board.height

        tent = board.last_last_change["last_last"]
        if tent == -1:
            return []
        i = board.move_to_location(tent)[0]
        j = board.move_to_location(tent)[1]

        player = array_2d[i][j]

        target30 = [0, player, player, player, 0]
        target31 = [0, player, 0, player, player, 0]
        target32 = [0, player, player, 0, player, 0]

        window_size1 = len(target30)
        window_size2 = len(target31)

        results = set()

        indexlist1 = list([i + 4, j + k + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A1 = list(board1[m[0], m[1]] for m in indexlist1)

        for a in range(len(A1) - window_size1 + 1):
            if A1[a:a + window_size1] == target30:
                results.add(board19[i + 4][j + 4 - (4 - a)])
                results.add(board19[i + 4][j + 4 + a])

        for a in range(len(A1) - window_size2 + 1):
            if A1[a:a + window_size2] == target31:
                results.add(board19[i + 4][j + 4 - (4 - a)])
                results.add(board19[i + 4][j + 4 - (2 - a)])
                results.add(board19[i + 4][j + 4 + (1 + a)])

            if A1[a:a + window_size2] == target32:
                results.add(board19[i + 4][j + 4 - (4 - a)])
                results.add(board19[i + 4][j + 4 - (1 - a)])
                results.add(board19[i + 4][j + 4 + (1 + a)])

        indexlist2 = list([i + k + 4, j + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A2 = list(board1[m[0], m[1]] for m in indexlist2)

        for a in range(len(A2) - window_size1 + 1):
            if A2[a:a + window_size1] == target30:
                results.add(board19[i + 4 - (4 - a)][j + 4])
                results.add(board19[i + 4 + a][j + 4])

        for a in range(len(A2) - window_size2 + 1):
            if A2[a:a + window_size2] == target31:
                results.add(board19[i + 4 - (4 - a)][j + 4])
                results.add(board19[i + 4 - (2 - a)][j + 4])
                results.add(board19[i + 4 + (1 + a)][j + 4])

            if A2[a:a + window_size2] == target32:
                results.add(board19[i + 4 - (4 - a)][j + 4])
                results.add(board19[i + 4 - (1 - a)][j + 4])
                results.add(board19[i + 4 + (1 + a)][j + 4])

        indexlist3 = list([i + k + 4, j + k + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A3 = list(board1[m[0], m[1]] for m in indexlist3)

        for a in range(len(A3) - window_size1 + 1):
            if A3[a:a + window_size1] == target30:
                results.add(board19[i + 4 - (4 - a)][j + 4 - (4 - a)])
                results.add(board19[i + 4 + a][j + 4 + a])

        for a in range(len(A3) - window_size2 + 1):
            if A3[a:a + window_size2] == target31:
                results.add(board19[i + 4 - (4 - a)][j + 4 - (4 - a)])
                results.add(board19[i + 4 - (2 - a)][j + 4 - (2 - a)])
                results.add(board19[i + 4 + (1 + a)][j + 4 + (1 + a)])

            if A3[a:a + window_size2] == target32:
                results.add(board19[i + 4 - (4 - a)][j + 4 - (4 - a)])
                results.add(board19[i + 4 - (1 - a)][j + 4 - (1 - a)])
                results.add(board19[i + 4 + (1 + a)][j + 4 + (1 + a)])

        indexlist4 = list([i + k + 4, j - k + 4] for k in (-4, -3, -2, -1, 0, 1, 2, 3, 4))
        A4 = list(board1[m[0], m[1]] for m in indexlist4)

        for a in range(len(A4) - window_size1 + 1):
            if A4[a:a + window_size1] == target30:
                results.add(board19[i + 4 - (4 - a)][j + 4 + (4 - a)])
                results.add(board19[i + 4 + a][j + 4 - a])

        for a in range(len(A4) - window_size2 + 1):
            if A4[a:a + window_size2] == target31:
                results.add(board19[i + 4 - (4 - a)][j + 4 + (4 - a)])
                results.add(board19[i + 4 - (2 - a)][j + 4 + (2 - a)])
                results.add(board19[i + 4 + (1 + a)][j + 4 - (1 + a)])

            if A4[a:a + window_size2] == target32:
                results.add(board19[i + 4 - (4 - a)][j + 4 + (4 - a)])
                results.add(board19[i + 4 - (1 - a)][j + 4 + (1 - a)])
                results.add(board19[i + 4 + (1 + a)][j + 4 - (1 + a)])

        return results

    def skipf(self, board):
        indic = self.checkai4(board)
        if len(indic) != 0:
            return list(indic)
        else:
            indic2 = self.checkp4(board)
            if len(indic2) != 0:
                return list(indic2)
            else:
                indic3 = self.checkp3(board)
                indic4 = self.checkai3(board)
                if len(indic3) != 0 and len(indic4) == 0:
                    return list(indic3)
                elif len(indic3) != 0 and len(indic4) != 0:
                    return list(indic3) + list(indic4)
                else:
                    fb = self.checkpforbid(board)
                    if len(fb) != 0 and len(indic4) != 0:
                        return list(fb) + list(indic4)
                    elif len(fb) != 0 and len(indic4) == 0:
                        return list(fb)
                    # elif len(fb) == 0 and len(indic4) != 0:
                    #     return list(indic4)
                    else:
                        return False

    def adjacent2(self, board):

        moved = list(set(range(board.width * board.height)) - set(board.availables))
        adjacents = set()
        width = board.width
        height = board.height

        # for m in moved:
        #     h = m // width + 4
        #     w = m % width + 4
        #     periphery = {board19[h+1][w], board19[h-1][w], board19[h][w+1], board19[h][w-1],
        #                  board19[h-1][w-1], board19[h-1][w+1], board19[h+1][w-1], board19[h+1][w+1]}
        #     adjacents.update(periphery)


        for m in moved:
            h = m // width
            w = m % width
            if w < width - 1:
                adjacents.add(m + 1)  # ��
            if w == width - 1:
                adjacents.add(m + 1 - width)  # �ҵ���
            if w > 0:
                adjacents.add(m - 1)  # ��
            if w == 0:
                adjacents.add(m - 1 + width)  # ����
            if h < height - 1:
                adjacents.add(m + width)  # ��
            if h == height - 1:
                adjacents.add(m + width - height * width)  # �µ���
            if h > 0:
                adjacents.add(m - width)  # ��
            if h == 0:
                adjacents.add(m - width + height * width)  # �ϵ���
            if w < width - 1 and h < height - 1:
                adjacents.add(m + width + 1)  # ����
            if w == width - 1 and h == height - 1:
                adjacents.add(m + width + 1 - width - height * width)  # ���µ�����
            if w > 0 and h < height - 1:
                adjacents.add(m + width - 1)  # ����
            if w == 0 and h == height - 1:
                adjacents.add(m + width - 1 + width - height * width)  # ���µ�����
            if w < width - 1 and h > 0:
                adjacents.add(m - width + 1)  # ����
            if w == width - 1 and h == 0:
                adjacents.add(m - width + 1 - width + height * width)  # ���ϵ�����
            if w > 0 and h > 0:
                adjacents.add(m - width - 1)  # ����
            if w == 0 and h == 0:
                adjacents.add(m - width - 1 + width + height * width)  # ���ϵ�����

        adjacents = list(set(adjacents) - set(moved))

        return adjacents

    def adjacent3(self, board):

        moved = list(set(range(board.width * board.height)) - set(board.availables))
        moved += self.adjacent2(board)
        adjacents = set()
        width = board.width
        height = board.height


        # for m in moved:
        #     h = m // width + 4
        #     w = m % width + 4
        #     periphery = {board19[h+1][w], board19[h-1][w], board19[h][w+1], board19[h][w-1],
        #                  board19[h-1][w-1], board19[h-1][w+1], board19[h+1][w-1], board19[h+1][w+1]}
        #     adjacents.update(periphery)

        for m in moved:
            h = m // width
            w = m % width
            if w < width - 1:
                adjacents.add(m + 1)  # ��
            if w == width - 1:
                adjacents.add(m + 1 - width)  # �ҵ���
            if w > 0:
                adjacents.add(m - 1)  # ��
            if w == 0:
                adjacents.add(m - 1 + width)  # ����
            if h < height - 1:
                adjacents.add(m + width)  # ��
            if h == height - 1:
                adjacents.add(m + width - height * width)  # �µ���
            if h > 0:
                adjacents.add(m - width)  # ��
            if h == 0:
                adjacents.add(m - width + height * width)  # �ϵ���
            if w < width - 1 and h < height - 1:
                adjacents.add(m + width + 1)  # ����
            if w == width - 1 and h == height - 1:
                adjacents.add(m + width + 1 - width - height * width)  # ���µ�����
            if w > 0 and h < height - 1:
                adjacents.add(m + width - 1)  # ����
            if w == 0 and h == height - 1:
                adjacents.add(m + width - 1 + width - height * width)  # ���µ�����
            if w < width - 1 and h > 0:
                adjacents.add(m - width + 1)  # ����
            if w == width - 1 and h == 0:
                adjacents.add(m - width + 1 - width + height * width)  # ���ϵ�����
            if w > 0 and h > 0:
                adjacents.add(m - width - 1)  # ����
            if w == 0 and h == 0:
                adjacents.add(m - width - 1 + width + height * width)  # ���ϵ�����

        adjacents = list(set(adjacents) - set(moved))

        return adjacents

    def checkpforbid(self, board):
        pool = self.adjacent2(board) + self.adjacent3(board)
        forbidmove = set()
        for i in pool:
            board_copy = copy.deepcopy(board)
            board_copy.update(-self.player, i)
            g3 = self.checkp3(board_copy)
            g4 = self.checkp4(board_copy)
            if len(g3) > 3 or len(g4) > 2 or (len(g4) > 0 and len(g3) > 0):
                forbidmove.add(i)

        return forbidmove



def main_template(player_is_black=True):
    
    global bline,row,col,board19
    bline = 11                  # the board size is 11x11 => need to draw 11 lines on the board
    
    pg.init()
    surface = draw_board()
    
    board = zmtBoard()
    
    board.init_board()
    
    running = True
    gameover = False


    draw_stone(surface, [5, 5], 1)
    board.update(1,60)
    
    row=None
    col=None
    
    colorskcai = -1 if player_is_black else 1
    skcplay_turn=[colorskcai,-colorskcai]
    zmtplay_turn=[-colorskcai,colorskcai]
    
    zmtAI=zmtMCTS(board,zmtplay_turn)
    skcAI=skcMCTS(board,skcplay_turn)
    
    
    array_key = np.array(range(121)).reshape(11, 11)  # ������ 0 - 128
    
    arraykey1 = np.concatenate((array_key[-4:,-4:], array_key[-4:,:],array_key[-4:,:4]), axis=1)
    arraykey2 = np.concatenate((array_key[:,-4:], array_key,array_key[:,:4]), axis=1)
    arraykey3 = np.concatenate((array_key[:4,-4:], array_key[:4,:],array_key[:4,:4]), axis=1)
    
    board19 = np.concatenate((arraykey1, arraykey2, arraykey3), axis=0)  # 19��19 ������ ������ 0 - 128
    
    if not player_is_black:
          color2 = 1 if player_is_black else -1
          zmtaimove = zmtAI.get_action()
          draw_stone(surface, board.move_to_location(zmtaimove), color2)
          board.update(color2, zmtaimove)        
    
    while running:
        
        for event in pg.event.get():              # A for loop to process all the events initialized by the player
             
            if event.type == pg.QUIT:             # terminate if player closes the game window 
                running = False
            
            
            color = -1 if player_is_black else 1
            skcaimove = skcAI.get_action()
            draw_stone(surface, board.move_to_location(skcaimove), color)
            board.update(color,skcaimove)

            
            if skcAI.check_winner(skcAI.board) != 0:
                print_winner(surface, winner=skcAI.check_winner(skcAI.board))

            else:
                color2 = 1 if player_is_black else -1
                zmtaimove = zmtAI.get_action()
                draw_stone(surface, board.move_to_location(zmtaimove), color2)
                board.update(color2, zmtaimove)
                if zmtAI.check_winner(zmtAI.board) != 0:
                    print_winner(surface, winner=zmtAI.check_winner(zmtAI.board))
        
        ####################################################################################################
        ######################## Normally Your edit should be within the while loop ########################
        ####################################################################################################

    
        
    pg.quit()


if __name__ == '__main__':
    main_template(True)     # zmt����Ϊtrue









