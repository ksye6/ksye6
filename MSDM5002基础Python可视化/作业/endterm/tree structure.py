#coding=utf-8

import numpy as np
import copy
import random
import time
import math
from collections import defaultdict
# board = np.zeros((11,11), dtype=int)
# print(board)

import pygame as pg
import random

class Board(object):
    """
    board for game
    """

    def __init__(self, width=11, height=11, n_in_row=5):
        self.width = width
        self.height = height
        self.states = {}  # 记录当前棋盘的状态，键是位置，值是棋子，这里用玩家来表示棋子类型
        self.last_change = {"last": -1}
        self.last_last_change = {"last_last": -1}
        self.n_in_row = n_in_row  # 表示几个相同的棋子连成一线算作胜利

    def init_board(self):
        self.availables = list(range(self.width * self.height))  # 表示棋盘上所有合法的位置，这里简单的认为空的位置即合法

        for m in self.availables:
            self.states[m] = 0  # 0表示当前位置为空

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

    def update(self, player, move):  # player在move处落子，更新棋盘
        self.states[move] = player
        self.availables.remove(move)
        self.last_last_change["last_last"] = self.last_change["last"]
        self.last_change["last"] = move

class TreeNode(object):
    def __init__(self, parent=None, pre_pos=None, board=None, player=None):
        self.pre_pos = pre_pos  # [0-121]    # 造成这个棋盘的结点下标

        self.parent = parent  # 父结点
        self.children = list()  # 子结点
        # self.player = player  # 下这步棋的玩家

        self.not_visit_pos = None  # 未访问过的节点

        self.board = board  # 每个结点对应一个棋盘状态

        self.num_of_visit = 0  # 访问次数N
        # self.num_of_win = 0                     # 胜利次数M 需要实时更新
        self.num_of_wins = defaultdict(int)  # 记录该结点模拟的白子、黑子的胜利次数(defaultdict: 当字典里的key不存在但被查找时，返回0)

    def fully_expanded(self,availables):
        """
        :return: True: 该结点已经完全扩展, False: 该结点未完全扩展
        """
        if self.not_visit_pos is None:  # 如果未访问过的结点为None(初始化为None)则未进行扩展过
            self.not_visit_pos = set(availables) - set([ele.pre_pos for ele in self.children]) # 得到可作为该结点扩展结点的所有下标

        return True if (len(self.not_visit_pos) == 0 and len(self.children) != 0) else False

    # def pick_univisted(self):
    #     """选择一个未访问的结点"""
    #     random_index = randint(0, len(self.not_visit_pos) - 1)  # 随机选择一个未访问的结点（random.randint: 闭区间）
    #     # print(len(self.not_visit_pos))
    #     move_pos = self.not_visit_pos.pop(random_index)  # 得到一个随机的未访问结点, 并从所有的未访问结点中删除
    #     # print(len(self.not_visit_pos))
    #
    #     new_board = self.board.move(move_pos)  # 模拟落子并返回新棋盘
    #     new_node = TreeNode(parent=self, pre_pos=move_pos, board=new_board)  # 新棋盘绑定新结点
    #     self.children.append(new_node)
    #     return new_node
    #
    # def pick_random(self):
    #     """选择结点的孩子进行扩展"""
    #     possible_moves = self.board.get_legal_pos()  # 可以落子的点位
    #     random_index = randint(0, len(possible_moves) - 1)  # 随机选择一个可以落子的点位（random.randint: 闭区间）
    #     move_pos = possible_moves[random_index]  # 得到一个随机的可以落子的点位
    #
    #     new_board = self.board.move(move_pos)  # 模拟落子并返回新棋盘
    #     new_node = TreeNode(parent=self, pre_pos=move_pos, board=new_board)  # 新棋盘绑定新结点
    #     return new_node

    # def non_terminal(self):
    #     """
    #     :return: None: 不是叶子(终端)结点, 'win' or 'tie': 是叶子(终端)结点
    #     """
    #     game_result = self.board.game_over(self.pre_pos)
    #     return game_result

    def num_of_win(self, playturn):
        # print('playturn:',playturn)
        wins = self.num_of_wins[playturn]  # 孩子结点的棋盘状态是在父节点的next_player之后形成
        # loses = self.num_of_wins[-playturn]
        return wins

    def best_uct(self,playturn, c_param=1.98):
        """返回一个自己最好的孩子结点（根据UCT进行比较）"""
        uct_of_children = np.array(list([
            (child.num_of_win(playturn) / child.num_of_visit) + c_param * np.sqrt(
                np.log(self.num_of_visit) / child.num_of_visit)
            for child in self.children
        ]))
        best_index = np.argmax(uct_of_children)
        # max_uct = max(uct_of_children)
        # best_index = np.where(uct_of_children == max_uct)     # 获取最大uct的下标
        # best_index = np.random.choice(best_index[0])        # 随机选取一个拥有最大uct的孩子
        return self.children[best_index]

class MCTS(object):
    """
    AI player, use Monte Carlo Tree Search with UCB
    """

    def __init__(self, board, play_turn, n_in_row=5, time=5, max_actions=1000):

        self.board = board
        self.play_turn = play_turn  # 出手顺序
        self.calculation_time = float(time)  # 最大运算时间
        self.max_actions = max_actions  # 每次模拟对局最多进行的步数
        self.n_in_row = n_in_row

        self.player = play_turn[0]  # 轮到电脑出手，所以出手顺序中第一个总是电脑
        self.confident = 1.96  # UCB中的常数 1.96
        # self.plays = {}  # 记录着法参与模拟的次数，键形如(player, move)，即（玩家，落子）
        # self.wins = {}  # 记录着法获胜的次数


        self.max_depth = 1
        self.nodenum = 0
        self.skip = False

    def get_action(self):  # return move

        if len(self.board.availables) == 1:
            return self.board.availables[0]  # 棋盘只剩最后一个落子位置，直接返回

        # 每次计算下一步时都要清空plays和wins表，因为经过AI和玩家的2步棋之后，整个棋盘的局面发生了变化，原来的记录已经不适用了——原先普通的一步现在可能是致胜的一步，如果不清空，会影响现在的结果，导致这一步可能没那么“致胜”了
        # self.plays = {}
        # self.wins = {}

        root = TreeNode() # 初始化根结点, 每次计算下一步都要重新初始化根结点
        self.nodenum = 0
        self.skip = False
        simulations = 0
        begin = time.time()
        while time.time() - begin < self.calculation_time:
            board_copy = copy.deepcopy(self.board)  # 模拟会修改board的参数，所以必须进行深拷贝，与原board进行隔离
            play_turn_copy = copy.deepcopy(self.play_turn)  # 每次模拟都必须按照固定的顺序进行，所以进行深拷贝防止顺序被修改
            self.run_simulation(board_copy, play_turn_copy,root)  # 进行MCTS
            simulations += 1
            # print('simulations:',simulations,'temp:',temp)


        print("total simulations=", simulations)

        self.skip = self.skipf(self.board)
        move, percent_wins = self.select_one_move(self.board,root)  # 选择最佳着法

        location = self.board.move_to_location(move)
        print('Maximum depth searched:', self.max_depth)
        print('Maximum node number:', self.nodenum)

        print("AI move: %d,%d\n" % (location[0], location[1]))
        print('AI move percent_wins: %f\n' % (percent_wins))

        return move


    def run_simulation(self, board, play_turn,root):
        """
        MCTS main process
        """

        # plays = self.plays
        # wins = self.wins

        node = root
        # visited = copy.deepcopy(root)

        # availables = board.availables

        player = self.get_player(play_turn)  # 获取当前出手的玩家
        # visited_states = set()  # 记录当前路径上的全部着法
        winner = 0
        expand = True

        # Simulation
        for t in range(1, self.max_actions + 1):
            # Selection
            # 如果所有着法都有统计信息，则获取UCB最大的着法
            availables = board.availables

            # if node.fully_expanded(availables):
            if len(set(availables) - set([ele.pre_pos for ele in node.children])) == 0:
                node = node.best_uct(player)
                # visited = visited.best_uct()
                move = node.pre_pos
                # print('move:',move)

            else:
                if expand:
                    plays = [ele.pre_pos for ele in node.children]
                else:
                    plays = []
                adjacents = []
                if len(availables) > self.n_in_row:
                    adjacents = self.adjacent_moves(board, player, plays)  # 没有统计信息的邻近位置

                if len(adjacents):
                    move = random.choice(adjacents)
                else:
                    peripherals = []
                    for move in availables:
                        if move not in plays:
                            peripherals.append(move)  # 没有统计信息的外围位置
                    move = random.choice(peripherals)

            board.update(player, move)

            # visited_new = TreeNode(parent=visited, pre_pos=move, board=board)
            # visited.children.append(visited_new)
            # visited = visited_new

            # Expand
            # 每次模拟最多扩展一次，每次扩展只增加一个着法
            if expand and move not in [ele.pre_pos for ele in node.children]:
                expand = False
                newnode = TreeNode(parent=node, pre_pos=move, board=board)
                node.children.append(newnode)
                node = newnode
                self.nodenum += 1
                if t > self.max_depth:
                    self.max_depth = t

            # visited_states.add((player, move))

            is_full = not len(availables)
            win = self.check_winner(board) != 0

            winner = 0
            if win:
                winner = self.check_winner(board)

            if is_full or win:  # 游戏结束，没有落子位置或有玩家获胜
                break

            player = self.get_player(play_turn)

        #bp
        # temp = node.pre_pos
        node.num_of_visit += 1
        node.num_of_wins[winner] += 1
        while node.parent:
            node = node.parent
            node.num_of_visit += 1
            node.num_of_wins[winner] += 1
        # return temp


    def get_player(self, players):
        p = players.pop(0)
        players.append(p)
        return p

    def select_one_move(self, board, node):
        if self.skip:
            limited = np.array(self.skip)
            visit_num_of_children = np.array(list([child.num_of_visit for child in node.children]))
            pre_pos_of_children = np.array(list([child.pre_pos for child in node.children]))
            idx = np.where(np.isin(pre_pos_of_children, limited))
            visit_num_of_children = visit_num_of_children[idx]

            best_index = np.argmax(visit_num_of_children)  # 获取最大ucb的下标
            node = np.array(node.children)[idx][best_index]
        else:
            limited = np.array(self.adjacent2(board)+self.adjacent3(board))
            # print('lim:',limited)

            visit_num_of_children = np.array(list([child.num_of_visit for child in node.children]))
            pre_pos_of_children = np.array(list([child.pre_pos for child in node.children]))

            # print('prepos:',pre_pos_of_children)
            # print('isin:',np.isin(pre_pos_of_children, limited))

            idx = np.where(np.isin(pre_pos_of_children, limited))
            visit_num_of_children = visit_num_of_children[idx]

            best_index = np.argmax(visit_num_of_children)  # 获取最大ucb的下标
            # print('idx:',idx,best_index)
            # print('node:',len(np.array(node.children)))
            node = np.array(node.children)[idx][best_index]
        # print('win:',node.num_of_win(self.play_turn[0]),'visit:',node.num_of_visit)
        return node.pre_pos, node.num_of_win(self.play_turn[0]) / node.num_of_visit

    def check_winner(self, board):
        """
        检查是否有玩家获胜
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)

        array11 = np.concatenate((array_2d[-4:, -4:], array_2d[-4:, :], array_2d[-4:, :4]), axis=1)
        array12 = np.concatenate((array_2d[:, -4:], array_2d, array_2d[:, :4]), axis=1)
        array13 = np.concatenate((array_2d[:4, -4:], array_2d[:4, :], array_2d[:4, :4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19×19 判定棋盘

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
        获取当前棋局中所有棋子的邻近位置中没有统计信息的位置
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
                adjacents.add(m + 1)  # 右
            if w == width - 1:
                adjacents.add(m + 1 - width)  # 右到左

            # if w <= width - 1:
            #     adjacents.add(m + 1 - width * (w//width)) # 右/右到左

            if w > 0:
                adjacents.add(m - 1)  # 左
            if w == 0:
                adjacents.add(m - 1 + width)  # 左到右
            if h < height - 1:
                adjacents.add(m + width)  # 下
            if h == height - 1:
                adjacents.add(m + width - height * width)  # 下到上
            if h > 0:
                adjacents.add(m - width)  # 上
            if h == 0:
                adjacents.add(m - width + height * width)  # 上到下
            if w < width - 1 and h < height - 1:
                adjacents.add(m + width + 1)  # 右下
            if w == width - 1 and h == height - 1:
                adjacents.add(m + width + 1 - width - height * width)  # 右下到左上
            if w > 0 and h < height - 1:
                adjacents.add(m + width - 1)  # 左下
            if w == 0 and h == height - 1:
                adjacents.add(m + width - 1 + width - height * width)  # 左下到右上
            if w < width - 1 and h > 0:
                adjacents.add(m - width + 1)  # 右上
            if w == width - 1 and h == 0:
                adjacents.add(m - width + 1 - width + height * width)  # 右上到左下
            if w > 0 and h > 0:
                adjacents.add(m - width - 1)  # 左上
            if w == 0 and h == 0:
                adjacents.add(m - width - 1 + width + height * width)  # 左上到右下

        adjacents = list(set(adjacents) - set(moved))
        for move in adjacents:
            if move in plays:
                adjacents.remove(move)
        return adjacents

    def checkp4(self, board):
        """
        检查玩家4
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # 值矩阵 -1,0,1

        array11 = np.concatenate((array_2d[-4:, -4:], array_2d[-4:, :], array_2d[-4:, :4]), axis=1)
        array12 = np.concatenate((array_2d[:, -4:], array_2d, array_2d[:, :4]), axis=1)
        array13 = np.concatenate((array_2d[:4, -4:], array_2d[:4, :], array_2d[:4, :4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19×19 判定棋盘 值矩阵 -1,0,1

        n = board.height

        tent = board.last_change["last"]
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
        检查ai4
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # 值矩阵 -1,0,1

        array11 = np.concatenate((array_2d[-4:, -4:], array_2d[-4:, :], array_2d[-4:, :4]), axis=1)
        array12 = np.concatenate((array_2d[:, -4:], array_2d, array_2d[:, :4]), axis=1)
        array13 = np.concatenate((array_2d[:4, -4:], array_2d[:4, :], array_2d[:4, :4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19×19 判定棋盘 值矩阵 -1,0,1

        n = board.height

        tent = board.last_last_change["last_last"]
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

    def checkp3(self, board):
        """
        检查玩家3
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # 值矩阵 -1,0,1

        array11 = np.concatenate((array_2d[-4:, -4:], array_2d[-4:, :], array_2d[-4:, :4]), axis=1)
        array12 = np.concatenate((array_2d[:, -4:], array_2d, array_2d[:, :4]), axis=1)
        array13 = np.concatenate((array_2d[:4, -4:], array_2d[:4, :], array_2d[:4, :4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19×19 判定棋盘 值矩阵 -1,0,1

        n = board.height

        tent = board.last_change["last"]
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
        检查玩家3
        """
        array_2d = np.array([board.states[key] for key in range(121)]).reshape(11, 11)  # 值矩阵 -1,0,1

        array11 = np.concatenate((array_2d[-4:, -4:], array_2d[-4:, :], array_2d[-4:, :4]), axis=1)
        array12 = np.concatenate((array_2d[:, -4:], array_2d, array_2d[:, :4]), axis=1)
        array13 = np.concatenate((array_2d[:4, -4:], array_2d[:4, :], array_2d[:4, :4]), axis=1)

        board1 = np.concatenate((array11, array12, array13), axis=0)  # 19×19 判定棋盘 值矩阵 -1,0,1

        n = board.height

        tent = board.last_last_change["last_last"]
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
                adjacents.add(m + 1)  # 右
            if w == width - 1:
                adjacents.add(m + 1 - width)  # 右到左
            if w > 0:
                adjacents.add(m - 1)  # 左
            if w == 0:
                adjacents.add(m - 1 + width)  # 左到右
            if h < height - 1:
                adjacents.add(m + width)  # 下
            if h == height - 1:
                adjacents.add(m + width - height * width)  # 下到上
            if h > 0:
                adjacents.add(m - width)  # 上
            if h == 0:
                adjacents.add(m - width + height * width)  # 上到下
            if w < width - 1 and h < height - 1:
                adjacents.add(m + width + 1)  # 右下
            if w == width - 1 and h == height - 1:
                adjacents.add(m + width + 1 - width - height * width)  # 右下到左上
            if w > 0 and h < height - 1:
                adjacents.add(m + width - 1)  # 左下
            if w == 0 and h == height - 1:
                adjacents.add(m + width - 1 + width - height * width)  # 左下到右上
            if w < width - 1 and h > 0:
                adjacents.add(m - width + 1)  # 右上
            if w == width - 1 and h == 0:
                adjacents.add(m - width + 1 - width + height * width)  # 右上到左下
            if w > 0 and h > 0:
                adjacents.add(m - width - 1)  # 左上
            if w == 0 and h == 0:
                adjacents.add(m - width - 1 + width + height * width)  # 左上到右下

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
                adjacents.add(m + 1)  # 右
            if w == width - 1:
                adjacents.add(m + 1 - width)  # 右到左
            if w > 0:
                adjacents.add(m - 1)  # 左
            if w == 0:
                adjacents.add(m - 1 + width)  # 左到右
            if h < height - 1:
                adjacents.add(m + width)  # 下
            if h == height - 1:
                adjacents.add(m + width - height * width)  # 下到上
            if h > 0:
                adjacents.add(m - width)  # 上
            if h == 0:
                adjacents.add(m - width + height * width)  # 上到下
            if w < width - 1 and h < height - 1:
                adjacents.add(m + width + 1)  # 右下
            if w == width - 1 and h == height - 1:
                adjacents.add(m + width + 1 - width - height * width)  # 右下到左上
            if w > 0 and h < height - 1:
                adjacents.add(m + width - 1)  # 左下
            if w == 0 and h == height - 1:
                adjacents.add(m + width - 1 + width - height * width)  # 左下到右上
            if w < width - 1 and h > 0:
                adjacents.add(m - width + 1)  # 右上
            if w == width - 1 and h == 0:
                adjacents.add(m - width + 1 - width + height * width)  # 右上到左下
            if w > 0 and h > 0:
                adjacents.add(m - width - 1)  # 左上
            if w == 0 and h == 0:
                adjacents.add(m - width - 1 + width + height * width)  # 左上到右下

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

        xcoords = [x for x in np.arange(x1, x2, dx if x1 < x2 else -dx)]
        ycoords = [y for y in np.arange(y1, y2, dy if y1 < y2 else -dy)]

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
    # play_turn=[colorai,-colorai]
    play_turn = [1,-1]
    
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









