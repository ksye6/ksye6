# -*- coding: gbk -*-
import numpy as np
import copy
import random
import time
import math

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

class Board(object):
    """
    board for game
    """
 
    def __init__(self, width=11, height=11, n_in_row=5):
        self.width = width
        self.height = height 
        self.states = {}
        self.last_change = {"last":-1}
        self.last_last_change = {"last_last":-1}
        self.n_in_row = n_in_row 
        self.steps = 0
 
    def init_board(self):
        self.availables = list(range(self.width * self.height))
 
        for m in self.availables:
            self.states[m] = 0 # 0表示空
 
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
 
    def update(self, player, move): # 更新棋盘
        self.states[move] = player
        self.availables.remove(move)
        self.last_last_change["last_last"] = self.last_change["last"]
        self.last_change["last"] = move
        self.steps += 1

class MCTS(object):
    """
    AI player, use Monte Carlo Tree Search with UCB
    """
 
    def __init__(self, board, play_turn, n_in_row=5, time=5, max_actions=3000):
 
        self.board = board
        self.play_turn = play_turn # 出手顺序
        self.calculation_time = float(time) # 最大运算时间
        self.max_actions = max_actions # 每次模拟最步数
        self.n_in_row = n_in_row
 
        self.player = play_turn[0] # 第一个总是电脑
        self.confident = 2.33 # UCB常数 1.96
        self.plays = {} # 记录模拟次数，键形如(player, move)
        self.wins = {} # 记录获胜次数
        self.max_depth = 1
        self.skip = False
 
    def get_action(self): # return move
 
        if len(self.board.availables) == 1:
            return self.board.availables[0] 
 
        # 重置plays和wins
        self.plays = {} 
        self.wins = {}
        self.skip = False
        simulations = 0
        begin = time.time()
        while time.time() - begin < self.calculation_time:
            board_copy = copy.deepcopy(self.board)  
            play_turn_copy = copy.deepcopy(self.play_turn) 
            self.run_simulation(board_copy, play_turn_copy) 
            simulations += 1
 
        # print("total simulations=", simulations)
        
        self.skip = self.skipf(self.board)
        move = self.select_one_move(self.board) # 落子
        
        location = self.board.move_to_location(move)
        # print('Maximum depth searched:', self.max_depth)
 
        # print("AI move: %d,%d\n" % (location[0], location[1]))
 
        return move
 
    def run_simulation(self, board, play_turn):
        """
        MCTS main process
        """
 
        plays = self.plays
        wins = self.wins
        availables = board.availables
 
        player = self.get_player(play_turn) # 获取当前出手的玩家
        visited_states = set() # 记录当前路径上的全部着法
        winner = 0
        expand = True
 
        # Simulation
        for t in range(1, self.max_actions + 1):
            # Selection
            # 如果所有着法都有统计信息，则获取UCB最大的着法

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
                    adjacents = self.adjacent_moves(board, player, plays) # 没有统计信息的邻近位置
                
                if len(adjacents):
                    move = random.choice(adjacents)
                else:
                    peripherals = []
                    for move in availables:
                        if not plays.get((player, move)):
                            peripherals.append(move) # 没有统计信息的其他位置
                    move = random.choice(peripherals) 
            
            board.update(player, move)
 
            # Expand
            # 每次模拟最多扩展一次，每次扩展只增加一个着法
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
            
            if is_full or win: # 没有落子位置或有玩家获胜
                break
 
            player = self.get_player(play_turn)
 
        # Back-propagation
        for player, move in visited_states:
            if (player, move) not in plays:
                continue
            plays[(player, move)] += 1 # 当前路径上所有着法的模拟次数加1
            if player == winner:
                wins[(player, move)] += 1 # 获胜玩家的所有着法的胜利次数加1
 
    def get_player(self, players):
        p = players.pop(0)
        players.append(p)
        return p
 
    def select_one_move(self, board):
        
        if self.skip:
            percent_wins, move = max(
                (self.wins.get((self.player, move), 0) /
                self.plays.get((self.player, move), 1),
                move)
                for move in self.skip)
        
        elif board.steps >9:
            limited = self.adjacent2(board)+self.adjacent3(board)
            percent_wins, move = max(
                (self.wins.get((self.player, move), 0) /
                 self.plays.get((self.player, move), 1),
                 move)
                for move in limited)
        
        else:
            limited = self.adjacent2(board)
            percent_wins, move = max(
                (self.wins.get((self.player, move), 0) /
                 self.plays.get((self.player, move), 1),
                 move)
                for move in limited)
        
        return move
    
    def check_winner(self, board):
        """
        检查是否有玩家获胜
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)

        array11 = np.concatenate((array_2d[-4:,-4:], array_2d[-4:,:],array_2d[-4:,:4]), axis=1)
        array12 = np.concatenate((array_2d[:,-4:], array_2d,array_2d[:,:4]), axis=1)
        array13 = np.concatenate((array_2d[:4,-4:], array_2d[:4,:],array_2d[:4,:4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19×19 判定棋盘

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
        
    def adjacent_moves(self, board, player, plays):
        """
        获取当前棋局中所有棋子的邻近位置中没有统计信息的位置
        """
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        adjacents = set()
        width = board.width
        height = board.height
     
        for m in moved:
            h = m // width
            w = m % width
            if w < width - 1:
                adjacents.add(m + 1) # 右
            if w == width - 1:
                adjacents.add(m + 1 - width) # 右到左  
            if w > 0:
                adjacents.add(m - 1) # 左
            if w == 0:
                adjacents.add(m - 1 + width) # 左到右
            if h < height - 1:
                adjacents.add(m + width) # 下
            if h == height - 1:
                adjacents.add(m + width - height*width) # 下到上
            if h > 0:
                adjacents.add(m - width) # 上
            if h == 0:
                adjacents.add(m - width + height*width) # 上到下
            if w < width - 1 and h < height - 1:
                adjacents.add(m + width + 1) # 右下
            if w == width - 1 and h == height - 1:
                adjacents.add(m + width + 1 - width - height*width) # 右下到左上
            if w > 0 and h < height - 1:
                adjacents.add(m + width - 1) # 左下
            if w == 0 and h == height - 1:
                adjacents.add(m + width - 1 + width - height*width) # 左下到右上
            if w < width - 1 and h > 0:
                adjacents.add(m - width + 1) # 右上
            if w == width - 1 and h == 0:
                adjacents.add(m - width + 1 - width + height*width) # 右上到左下
            if w > 0 and h > 0:
                adjacents.add(m - width - 1) # 左上
            if w == 0 and h == 0:
                adjacents.add(m - width - 1 + width + height*width) # 左上到右下
     
        adjacents = list(set(adjacents) - set(moved))
        for move in adjacents:
            if plays.get((player, move)):
                adjacents.remove(move)
        return adjacents

    def checkp4(self, board):
        """
        检查玩家4
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # 值矩阵 -1,0,1
    
        array11 = np.concatenate((array_2d[-4:,-4:], array_2d[-4:,:],array_2d[-4:,:4]), axis=1)
        array12 = np.concatenate((array_2d[:,-4:], array_2d,array_2d[:,:4]), axis=1)
        array13 = np.concatenate((array_2d[:4,-4:], array_2d[:4,:],array_2d[:4,:4]), axis=1)
    
        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19×19 判定棋盘 值矩阵 -1,0,1
        
        n=board.height
        
        tent=board.last_change["last"]
        
        if tent == -1:
            return []
        
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
        检查ai4
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # 值矩阵 -1,0,1
    
        array11 = np.concatenate((array_2d[-4:,-4:], array_2d[-4:,:],array_2d[-4:,:4]), axis=1)
        array12 = np.concatenate((array_2d[:,-4:], array_2d,array_2d[:,:4]), axis=1)
        array13 = np.concatenate((array_2d[:4,-4:], array_2d[:4,:],array_2d[:4,:4]), axis=1)
    
        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19×19 判定棋盘 值矩阵 -1,0,1
        
        n=board.height
        
        tent=board.last_last_change["last_last"]
        
        if tent == -1:
            return []
            
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
        检查玩家3
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # 值矩阵 -1,0,1

        array11 = np.concatenate((array_2d[-4:,-4:], array_2d[-4:,:],array_2d[-4:,:4]), axis=1)
        array12 = np.concatenate((array_2d[:,-4:], array_2d,array_2d[:,:4]), axis=1)
        array13 = np.concatenate((array_2d[:4,-4:], array_2d[:4,:],array_2d[:4,:4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19×19 判定棋盘 值矩阵 -1,0,1

        n=board.height

        tent=board.last_change["last"]
        
        if tent == -1:
            return []
                
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
        检查AI3
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # 值矩阵 -1,0,1

        array11 = np.concatenate((array_2d[-4:,-4:], array_2d[-4:,:],array_2d[-4:,:4]), axis=1)
        array12 = np.concatenate((array_2d[:,-4:], array_2d,array_2d[:,:4]), axis=1)
        array13 = np.concatenate((array_2d[:4,-4:], array_2d[:4,:],array_2d[:4,:4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19×19 判定棋盘 值矩阵 -1,0,1

        n=board.height

        tent=board.last_last_change["last_last"]
        
        if tent == -1:
            return []
        
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
                # results.add(board19[i+4][j+4-(4-a)])
                results.add(board19[i+4][j+4-(2-a)])
                # results.add(board19[i+4][j+4+(1+a)])
            
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
                # results.add(board19[i+4-(4-a)][j+4])
                results.add(board19[i+4-(2-a)][j+4])
                # results.add(board19[i+4+(1+a)][j+4])
            
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
                # results.add(board19[i+4-(4-a)][j+4-(4-a)])
                results.add(board19[i+4-(2-a)][j+4-(2-a)])
                # results.add(board19[i+4+(1+a)][j+4+(1+a)])
            
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
                # results.add(board19[i+4-(4-a)][j+4+(4-a)])
                results.add(board19[i+4-(2-a)][j+4+(2-a)])
                # results.add(board19[i+4+(1+a)][j+4-(1+a)])
            
            if A4[a:a+window_size2] == target32:
                results.add(board19[i+4-(4-a)][j+4+(4-a)])
                results.add(board19[i+4-(1-a)][j+4+(1-a)])
                results.add(board19[i+4+(1+a)][j+4-(1+a)])

        return results

    def skipf(self, board):
        indic = self.checkai4(board) # ai4
        if len(indic) != 0:  # ai有4
            return list(indic)  # 直接下
        else:   # ai没有4
            indic2 = self.checkp4(board)  # 玩家4
            if len(indic2) != 0:  # 玩家有4
                return list(indic2)   # 堵玩家
            else:  # 玩家没有4
                indic3 = self.checkp3(board)  # 玩家活3
                indic4 = self.checkai3(board)  # ai活3
                if len(indic4) != 0:  # ai有活3
                    return list(indic4)  # 直接下
                elif len(indic3) != 0:  # ai没有活3，玩家有活3
                    return list(indic3)  # 堵玩家
                else:  # ai没有活3，玩家没有活3
                    # 仅考虑玩家或ai一次不产生多个禁手点
                    fbp = self.checkpforbidp(board)  # 玩家禁手
                    fbai = self.checkpforbidai(board)  # ai禁手
                    if len(fbp[0]) != 0 and len(fbai[0]) == 0: # 玩家有禁手，ai无禁手
                        return list(fbp[0]) # 堵玩家
                    elif len(fbp[0]) == 0 and len(fbai[0]) != 0: # 玩家无禁手，ai有禁手
                        return list(fbai[0]) # 走禁手
                    elif len(fbp[0]) != 0 and len(fbai[0]) != 0: # 均有禁手
                        if 'strong' == fbp[1][0] and 'weak' == fbai[1][0]:  # 玩家强禁手，ai弱禁手
                            return list(fbp[0]) # 堵玩家
                        else: # 其余情况
                            return list(fbai[0]) # 走ai禁手
                    else: # 均无禁手
                        pt = self.check_check_fbai(board) # ai潜力
                        if len(pt) != 0: # ai有潜力点
                            return list(pt) # 走潜力点
                        else:
                            return False
    
    def adjacent2(self, board):  # 周围一圈
      
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        adjacents = set()
        width = board.width
        height = board.height
     
        for m in moved:
            h = m // width
            w = m % width
            if w < width - 1:
                adjacents.add(m + 1) # 右
            if w == width - 1:
                adjacents.add(m + 1 - width) # 右到左  
            if w > 0:
                adjacents.add(m - 1) # 左
            if w == 0:
                adjacents.add(m - 1 + width) # 左到右
            if h < height - 1:
                adjacents.add(m + width) # 下
            if h == height - 1:
                adjacents.add(m + width - height*width) # 下到上
            if h > 0:
                adjacents.add(m - width) # 上
            if h == 0:
                adjacents.add(m - width + height*width) # 上到下
            if w < width - 1 and h < height - 1:
                adjacents.add(m + width + 1) # 右下
            if w == width - 1 and h == height - 1:
                adjacents.add(m + width + 1 - width - height*width) # 右下到左上
            if w > 0 and h < height - 1:
                adjacents.add(m + width - 1) # 左下
            if w == 0 and h == height - 1:
                adjacents.add(m + width - 1 + width - height*width) # 左下到右上
            if w < width - 1 and h > 0:
                adjacents.add(m - width + 1) # 右上
            if w == width - 1 and h == 0:
                adjacents.add(m - width + 1 - width + height*width) # 右上到左下
            if w > 0 and h > 0:
                adjacents.add(m - width - 1) # 左上
            if w == 0 and h == 0:
                adjacents.add(m - width - 1 + width + height*width) # 左上到右下
     
        adjacents = list(set(adjacents) - set(moved))

        return adjacents

    def adjacent3(self, board): # 周围第二圈
      
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        moved += self.adjacent2(board)
        adjacents = set()
        width = board.width
        height = board.height
     
        for m in moved:
            h = m // width
            w = m % width
            if w < width - 1:
                adjacents.add(m + 1) # 右
            if w == width - 1:
                adjacents.add(m + 1 - width) # 右到左  
            if w > 0:
                adjacents.add(m - 1) # 左
            if w == 0:
                adjacents.add(m - 1 + width) # 左到右
            if h < height - 1:
                adjacents.add(m + width) # 下
            if h == height - 1:
                adjacents.add(m + width - height*width) # 下到上
            if h > 0:
                adjacents.add(m - width) # 上
            if h == 0:
                adjacents.add(m - width + height*width) # 上到下
            if w < width - 1 and h < height - 1:
                adjacents.add(m + width + 1) # 右下
            if w == width - 1 and h == height - 1:
                adjacents.add(m + width + 1 - width - height*width) # 右下到左上
            if w > 0 and h < height - 1:
                adjacents.add(m + width - 1) # 左下
            if w == 0 and h == height - 1:
                adjacents.add(m + width - 1 + width - height*width) # 左下到右上
            if w < width - 1 and h > 0:
                adjacents.add(m - width + 1) # 右上
            if w == width - 1 and h == 0:
                adjacents.add(m - width + 1 - width + height*width) # 右上到左下
            if w > 0 and h > 0:
                adjacents.add(m - width - 1) # 左上
            if w == 0 and h == 0:
                adjacents.add(m - width - 1 + width + height*width) # 左上到右下
     
        adjacents = list(set(adjacents) - set(moved))

        return adjacents

    def checkpforbidp(self, board):
        """
        检查玩家禁手
        """ 
        pool = self.adjacent2(board)+self.adjacent3(board)
        forbidmove = set()
        attribute = list()
        for i in pool:
            board_copy = copy.deepcopy(board)
            board_copy.update(-self.player, i)
            g3 = self.checkp3(board_copy)
            g4 = self.checkp4(board_copy)
            if len(g3) > 3:
                forbidmove.add(i)
                attribute.append('weak')
            elif len(g4) >= 2 or (len(g4) > 0 and len(g3) > 0):
                forbidmove.add(i)
                attribute.append('strong')
        
        return forbidmove,attribute
    
    def checkpforbidai(self, board):
        """
        检查AI禁手
        """        
        pool = self.adjacent2(board)+self.adjacent3(board)
        forbidmove = set()
        attribute = list()
        for i in pool:
            board_copy = copy.deepcopy(board)
            board_copy.update(self.player, i)
            board_copy.last_last_change["last_last"]=board_copy.last_change["last"]
            g3 = self.checkai3_all(board_copy)
            g4 = self.checkai4(board_copy)
            if len(g3) > 3:
                forbidmove.add(i)
                attribute.append('weak')
                # print('find forbid for ai!!')
                # print('forbid:',self.player, 'g3:',g3, 'g4:',g4)
            elif len(g4) >= 2 or (len(g4) > 0 and len(g3) > 0):
                forbidmove.add(i)
                attribute.append('strong')
                # print('find forbid for ai!!')
                # print('forbid:',self.player, 'g3:',g3, 'g4:',g4)
        
        return forbidmove,attribute
        
    def checkai3_all(self, board):
        """
        检查AI3，返回所有点
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # 值矩阵 -1,0,1

        array11 = np.concatenate((array_2d[-4:,-4:], array_2d[-4:,:],array_2d[-4:,:4]), axis=1)
        array12 = np.concatenate((array_2d[:,-4:], array_2d,array_2d[:,:4]), axis=1)
        array13 = np.concatenate((array_2d[:4,-4:], array_2d[:4,:],array_2d[:4,:4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19×19 判定棋盘 值矩阵 -1,0,1

        n=board.height

        tent=board.last_last_change["last_last"]
        
        if tent == -1:
            return []
        
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
        
    def check_check_fbai(self, board):
        pool = self.adjacent2(board)+self.adjacent3(board)
        potential = set()
        for i in pool:
            board_copy = copy.deepcopy(board)
            board_copy.update(self.player, i)
            if len(self.checkpforbidai(board_copy)[0]) != 0:
                potential.add(i)
        
        return potential

def ai_move(B,c):
    global bline,row,col,board19,B_lastlast
    
    bline = 11
    row=None
    col=None
    if np.array_equal(np.zeros((11, 11), dtype=int),B):  #若棋盘B全空则下中间
        return 5,5
    
    
    #若第一次运行这个函数，创建一个B_lastlast全局变量初始化为全0，用来记录两回合前的状态。先手则第3回合B比B_lastlast多了一白一黑
    #，579...回合全部多一白一黑；后手则第2回合B比B_lastlast只多了一黑，468...回合全部多一黑一白
    if not hasattr(ai_move, 'is_first_run'):  
        B_lastlast=np.zeros((11, 11), dtype=int)
        ai_move.is_first_run = True
    
    #每次初始化类
    BOARD = Board()
    BOARD.init_board()
    play_turn=[c,-c]
    AI=MCTS(BOARD,play_turn)
    
    array_key = np.array(range(121)).reshape(11, 11)  # 键矩阵 0 - 128
    
    arraykey1 = np.concatenate((array_key[-4:,-4:], array_key[-4:,:],array_key[-4:,:4]), axis=1)
    arraykey2 = np.concatenate((array_key[:,-4:], array_key,array_key[:,:4]), axis=1)
    arraykey3 = np.concatenate((array_key[:4,-4:], array_key[:4,:],array_key[:4,:4]), axis=1)
    
    board19 = np.concatenate((arraykey1, arraykey2, arraykey3), axis=0)  # 19×19 键棋盘 键矩阵 0 - 128
    
    #先更新B上上回合的所有状态
    for i in range(11):
        for j in range(11):
            if B[i,j]!=0 and B_lastlast[i,j]==B[i,j]:
                BOARD.update(B[i,j],array_key[i,j])
    
    #再更新B上上回合自己走的，后手第一次落子则不需要更新
    for i in range(11):
        for j in range(11):
            if B_lastlast[i,j]!=B[i,j] and B[i,j]==c:
                BOARD.update(B[i,j],array_key[i,j])
    
    #再更新B上回合对面走的
    for i in range(11):
        for j in range(11):
            if B_lastlast[i,j]!=B[i,j] and B[i,j]==-c:
                BOARD.update(B[i,j],array_key[i,j])    
    
    aimove = AI.get_action() # 先判断走哪步
    aimove_index = BOARD.move_to_location(aimove) #变为坐标
    B_lastlast = B.copy()  # 再更新全局变量B_lastlast，用于两回合后
    return aimove_index


def main_template(player_is_black=True):
    
    global bline,row,col,board19
    bline = 11                  # the board size is 11x11 => need to draw 11 lines on the board
    
    pg.init()
    surface = draw_board()
    
    board = Board()
    
    board.init_board()
    
    running = True
    gameover = False

    if not player_is_black:
        draw_stone(surface, [5, 5], 1)
        board.update(1,60)
    
    row=None
    col=None
    
    colorai = -1 if player_is_black else 1
    play_turn=[colorai,-colorai]
    
    AI=MCTS(board,play_turn)
    
    array_key = np.array(range(121)).reshape(11, 11)  # 键矩阵 0 - 128
    
    arraykey1 = np.concatenate((array_key[-4:,-4:], array_key[-4:,:],array_key[-4:,:4]), axis=1)
    arraykey2 = np.concatenate((array_key[:,-4:], array_key,array_key[:,:4]), axis=1)
    arraykey3 = np.concatenate((array_key[:4,-4:], array_key[:4,:],array_key[:4,:4]), axis=1)
    
    board19 = np.concatenate((arraykey1, arraykey2, arraykey3), axis=0)  # 19×19 键棋盘 键矩阵 0 - 128
    
    while running:
        
        for event in pg.event.get():              # A for loop to process all the events initialized by the player
             
            if event.type == pg.QUIT:             # terminate if player closes the game window 
                running = False
                
            if event.type == pg.MOUSEBUTTONDOWN and not gameover:        # detect whether the player is clicking in the window
                
                (x,y) = event.pos                                        # check if the clicked position is on the 11x11 center grid
                if (x > pad+3.75*sep) and (x < w_size-pad-3.75*sep) and (y > pad+3.75*sep) and (y < w_size-pad-3.75*sep):
                    row = round((x-pad)/sep-4)     
                    col = round((y-pad)/sep-4)
                    move = row * bline + col
                    
                    if board.states[move] == 0:                             # update the board matrix if that position has not been occupied
                        color = 1 if player_is_black else -1
                        board.update(color,move)
                        draw_stone(surface, [row, col], color)
                        
                        if AI.check_winner(AI.board) != 0:
                            print_winner(surface, winner=AI.check_winner(AI.board))
        
                        else:
                            color2 = -1 if player_is_black else 1
                            aimove = AI.get_action()
                            draw_stone(surface, board.move_to_location(aimove), color2)
                            board.update(color2, aimove)
                            if AI.check_winner(AI.board) != 0:
                                print_winner(surface, winner=AI.check_winner(AI.board))
        
        ####################################################################################################
        ######################## Normally Your edit should be within the while loop ########################
        ####################################################################################################

    
        
    pg.quit()




if __name__ == '__main__':
    main_template(False)















