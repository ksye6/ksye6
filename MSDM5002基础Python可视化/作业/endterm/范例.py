class Board(object):
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

import random

class MCTS(object):
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
        self.confident = 2.33 # UCB�еĳ��� 1.96
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
                for move in limited) # ѡ��ʤ����ߵ��ŷ� # self.board.availables
        
        else:
            limited = self.adjacent2(board)
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
        ���ai4
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # ֵ���� -1,0,1
    
        array11 = np.concatenate((array_2d[-4:,-4:], array_2d[-4:,:],array_2d[-4:,:4]), axis=1)
        array12 = np.concatenate((array_2d[:,-4:], array_2d,array_2d[:,:4]), axis=1)
        array13 = np.concatenate((array_2d[:4,-4:], array_2d[:4,:],array_2d[:4,:4]), axis=1)
    
        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19��19 �ж����� ֵ���� -1,0,1
        
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
        ������3
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # ֵ���� -1,0,1

        array11 = np.concatenate((array_2d[-4:,-4:], array_2d[-4:,:],array_2d[-4:,:4]), axis=1)
        array12 = np.concatenate((array_2d[:,-4:], array_2d,array_2d[:,:4]), axis=1)
        array13 = np.concatenate((array_2d[:4,-4:], array_2d[:4,:],array_2d[:4,:4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19��19 �ж����� ֵ���� -1,0,1

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
        ���AI3
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # ֵ���� -1,0,1

        array11 = np.concatenate((array_2d[-4:,-4:], array_2d[-4:,:],array_2d[-4:,:4]), axis=1)
        array12 = np.concatenate((array_2d[:,-4:], array_2d,array_2d[:,:4]), axis=1)
        array13 = np.concatenate((array_2d[:4,-4:], array_2d[:4,:],array_2d[:4,:4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19��19 �ж����� ֵ���� -1,0,1

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
        if len(indic) != 0:  # ai��4
            return list(indic)  # ֱ����
        else:   # aiû��4
            indic2 = self.checkp4(board)  # ���4
            if len(indic2) != 0:  # �����4
                return list(indic2)   # �����
            else:  # ���û��4
                indic3 = self.checkp3(board)  # ��һ�3
                indic4 = self.checkai3(board)  # ai��3
                if len(indic4) != 0:  # ai�л�3
                    return list(indic4)  # ֱ����
                elif len(indic3) != 0:  # aiû�л�3������л�3
                    return list(indic3)  # �����
                else:  # aiû�л�3�����û�л�3
                    # ��������һ�aiһ�β�����������ֵ�
                    fbp = self.checkpforbidp(board)  # ��ҽ���
                    fbai = self.checkpforbidai(board)  # ai����
                    if len(fbp[0]) != 0 and len(fbai[0]) == 0: # ����н��֣�ai�޽���
                        return list(fbp[0]) # �����
                    elif len(fbp[0]) == 0 and len(fbai[0]) != 0: # ����޽��֣�ai�н���
                        return list(fbai[0]) # �߽���
                    elif len(fbp[0]) != 0 and len(fbai[0]) != 0: # ���н���
                        if 'strong' == fbp[1][0] and 'weak' == fbai[1][0]:  # ���ǿ���֣�ai������
                            return list(fbp[0]) # �����
                        else: # �������
                            return list(fbai[0]) # ��ai����
                    else: # ���޽���
                        pt = self.check_check_fbai(board) # aiǱ��
                        if len(pt) != 0: # ai��Ǳ����
                            return list(pt) # ��Ǳ����
                        else:
                            return False
    
    def adjacent2(self, board):  # ��ΧһȦ
      
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

    def adjacent3(self, board): # ��Χ�ڶ�Ȧ
      
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

    def checkpforbidp(self, board):
        """
        �����ҽ���
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
        ���AI����
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
        ���AI3���������е�
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # ֵ���� -1,0,1

        array11 = np.concatenate((array_2d[-4:,-4:], array_2d[-4:,:],array_2d[-4:,:4]), axis=1)
        array12 = np.concatenate((array_2d[:,-4:], array_2d,array_2d[:,:4]), axis=1)
        array13 = np.concatenate((array_2d[:4,-4:], array_2d[:4,:],array_2d[:4,:4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19��19 �ж����� ֵ���� -1,0,1

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
    
            
            
            
            
        
        
