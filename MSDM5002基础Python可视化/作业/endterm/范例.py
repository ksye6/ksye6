class Board(object):
    """
    board for game
    """
 
    def __init__(self, width=11, height=11, n_in_row=5):
        self.width = width
        self.height = height 
        self.states = {} # 记录当前棋盘的状态，键是位置，值是棋子，这里用玩家来表示棋子类型
        self.n_in_row = n_in_row # 表示几个相同的棋子连成一线算作胜利
 
    def init_board(self):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not less than %d' % self.n_in_row) # 棋盘不能过小
 
        self.availables = list(range(self.width * self.height)) # 表示棋盘上所有合法的位置，这里简单的认为空的位置即合法
 
        for m in self.availables:
            self.states[m] = -1 # -1表示当前位置为空
 
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



class MCTS(object):

    def __init__(self, board, play_turn, n_in_row=5, time=10, max_actions=1000):
 
        self.board = board
        self.play_turn = play_turn # 出手顺序
        self.calculation_time = float(time) # 最大运算时间
        self.max_actions = max_actions # 每次模拟对局最多进行的步数
        self.n_in_row = n_in_row
 
        self.player = play_turn[0] # 轮到电脑出手，所以出手顺序中第一个总是电脑
        self.confident = 1.96 # UCB中的常数
        self.plays = {} # 记录着法参与模拟的次数，键形如(player, move)，即（玩家，落子）
        self.wins = {} # 记录着法获胜的次数
        self.max_depth = 1
 
    def get_action(self): # return move
 
        if len(self.board.availables) == 1:
            return self.board.availables[0] # 棋盘只剩最后一个落子位置，直接返回
 
        # 每次计算下一步时都要清空plays和wins表，因为经过AI和玩家的2步棋之后，整个棋盘的局面发生了变化，原来的记录已经不适用了——原先普通的一步现在可能是致胜的一步，如果不清空，会影响现在的结果，导致这一步可能没那么“致胜”了
        self.plays = {} 
        self.wins = {}
        simulations = 0
        begin = time.time()
        while time.time() - begin < self.calculation_time:
            board_copy = copy.deepcopy(self.board)  # 模拟会修改board的参数，所以必须进行深拷贝，与原board进行隔离
            play_turn_copy = copy.deepcopy(self.play_turn) # 每次模拟都必须按照固定的顺序进行，所以进行深拷贝防止顺序被修改
            self.run_simulation(board_copy, play_turn_copy) # 进行MCTS
            simulations += 1
 
        print("total simulations=", simulations)
 
        move = self.select_one_move() # 选择最佳着法
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
        winner = -1
        expand = True
 
        # Simulation
        for t in range(1, self.max_actions + 1):
            # Selection
            # 如果所有着法都有统计信息，则获取UCB最大的着法
            if all(plays.get((player, move)) for move in availables):
                log_total = log(
                    sum(plays[(player, move)] for move in availables))
                value, move = max(
                    ((wins[(player, move)] / plays[(player, move)]) +
                     sqrt(self.confident * log_total / plays[(player, move)]), move)
                    for move in availables) 
            else:
                adjacents = []
                if len(availables) > self.n_in_row:
                    adjacents = self.adjacent_moves(board, player, plays) # 没有统计信息的邻近位置
     
                if len(adjacents):
                    move = choice(adjacents)
                else:
                    peripherals = []
                    for move in availables:
                        if not plays.get((player, move)):
                            peripherals.append(move) # 没有统计信息的外围位置
                    move = choice(peripherals)
            
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
            win, winner = self.has_a_winner(board)
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
 
    def select_one_move(self):
        percent_wins, move = max(
            (self.wins.get((self.player, move), 0) /
             self.plays.get((self.player, move), 1),
             move)
            for move in self.board.availables) # 选择胜率最高的着法
 
        return move
 
    def has_a_winner(self, board):
        """
        检查是否有玩家获胜
        """
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        if(len(moved) < self.n_in_row + 2):
            return False, -1

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
            if w > 0:
                adjacents.add(m - 1) # 左
            if h < height - 1:
                adjacents.add(m + width) # 上
            if h > 0:
                adjacents.add(m - width) # 下
            if w < width - 1 and h < height - 1:
                adjacents.add(m + width + 1) # 右上
            if w > 0 and h < height - 1:
                adjacents.add(m + width - 1) # 左上
            if w < width - 1 and h > 0:
                adjacents.add(m - width + 1) # 右下
            if w > 0 and h > 0:
                adjacents.add(m - width - 1) # 左下
     
        adjacents = list(set(adjacents) - set(moved))
        for move in adjacents:
            if plays.get((player, move)):
                adjacents.remove(move)
        return adjacents
















