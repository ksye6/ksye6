class Board(object):
    """
    board for game
    """
 
    def __init__(self, width=11, height=11, n_in_row=5):
        self.width = width
        self.height = height 
        self.states = {} # 记录当前棋盘的状态，键是位置，值是棋子，这里用玩家来表示棋子类型
        self.last_change = {"last":-1}
        self.last_last_change = {"last_last":-1}
        self.n_in_row = n_in_row # 表示几个相同的棋子连成一线算作胜利
        self.steps = 0
 
    def init_board(self):
        self.availables = list(range(self.width * self.height)) # 表示棋盘上所有合法的位置，这里简单的认为空的位置即合法
 
        for m in self.availables:
            self.states[m] = 0 # 0表示当前位置为空
 
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
 
    def update(self, player, move): # player在move处落子，更新棋盘
        self.states[move] = player
        self.availables.remove(move)
        self.last_last_change["last_last"] = self.last_change["last"]
        self.last_change["last"] = move
        self.steps += 1

import random

class MCTS(object):
    """
    AI player, use Monte Carlo Tree Search with UCB
    """
 
    def __init__(self, board, play_turn, n_in_row=5, time=5, max_actions=3000):
 
        self.board = board
        self.play_turn = play_turn # 出手顺序
        self.calculation_time = float(time) # 最大运算时间
        self.max_actions = max_actions # 每次模拟对局最多进行的步数
        self.n_in_row = n_in_row
 
        self.player = play_turn[0] # 轮到电脑出手，所以出手顺序中第一个总是电脑
        self.confident = 2.33 # UCB中的常数 1.96
        self.plays = {} # 记录着法参与模拟的次数，键形如(player, move)，即（玩家，落子）
        self.wins = {} # 记录着法获胜的次数
        self.max_depth = 1
        self.skip = False
 
    def get_action(self): # return move
 
        if len(self.board.availables) == 1:
            return self.board.availables[0] # 棋盘只剩最后一个落子位置，直接返回
 
        # 每次计算下一步时都要清空plays和wins表，因为经过AI和玩家的2步棋之后，整个棋盘的局面发生了变化，原来的记录已经不适用了——原先普通的一步现在可能是致胜的一步，如果不清空，会影响现在的结果，导致这一步可能没那么“致胜”了
        self.plays = {} 
        self.wins = {}
        self.skip = False
        simulations = 0
        begin = time.time()
        while time.time() - begin < self.calculation_time:
            board_copy = copy.deepcopy(self.board)  # 模拟会修改board的参数，所以必须进行深拷贝，与原board进行隔离
            play_turn_copy = copy.deepcopy(self.play_turn) # 每次模拟都必须按照固定的顺序进行，所以进行深拷贝防止顺序被修改
            self.run_simulation(board_copy, play_turn_copy) # 进行MCTS
            simulations += 1
 
        print("total simulations=", simulations)
        
        self.skip = self.skipf(self.board)
        move = self.select_one_move(self.board) # 选择最佳着法
        
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
                            peripherals.append(move) # 没有统计信息的外围位置
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
            
            if is_full or win: # 游戏结束，没有落子位置或有玩家获胜
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
        
        elif board.steps >10:
            limited = self.adjacent2(board)+self.adjacent3(board)
            percent_wins, move = max(
                (self.wins.get((self.player, move), 0) /
                 self.plays.get((self.player, move), 1),
                 move)
                for move in limited) # 选择胜率最高的着法 # self.board.availables
        
        else:
            limited = self.adjacent2(board)
            percent_wins, move = max(
                (self.wins.get((self.player, move), 0) /
                 self.plays.get((self.player, move), 1),
                 move)
                for move in limited) # 选择胜率最高的着法 # self.board.availables
        
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
        #         len(set(states[i] for i in range(m, m + n))) == 1): # 横向连成一线
        #         return player
        # 
        #     if (h in range(height - n + 1) and
        #         len(set(states[i] for i in range(m, m + n * width, width))) == 1): # 竖向连成一线
        #         return player
        # 
        #     if (w in range(width - n + 1) and h in range(height - n + 1) and
        #         len(set(states[i] for i in range(m, m + n * (width + 1), width + 1))) == 1): # 右斜向上连成一线
        #         return player
        # 
        #     if (w in range(n - 1, width) and h in range(height - n + 1) and
        #         len(set(states[i] for i in range(m, m + n * (width - 1), width - 1))) == 1): # 左斜向下连成一线
        #         return player
        # 
        # if count == width*height:
        #     return 2
        # 
        # return 0
    
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
                print('find forbid for ai!!')
                print('forbid:',self.player, 'g3:',g3, 'g4:',g4)
            elif len(g4) >= 2 or (len(g4) > 0 and len(g3) > 0):
                forbidmove.add(i)
                attribute.append('strong')
                print('find forbid for ai!!')
                print('forbid:',self.player, 'g3:',g3, 'g4:',g4)
        
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
    
            
            
            
            
        
        
