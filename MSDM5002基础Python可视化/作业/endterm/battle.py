# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:36:42 2023

@author: mings
"""

# change "group_test" to your filename or folder name
# if this code can run then you are following my instruction perfectly
import sys
module_path = 'C:\\Users\\张铭韬\\Desktop\\学业\\港科大\\MSDM5002基础Python可视化\\作业\\endterm'
sys.path.append(module_path)
import importlib
importlib.reload(gp1)

import final as gp1

import pygame as pg
import numpy as np
import utility as u


def battle(ai_move1, ai_move2):
    
    pg.init()
    surface = u.draw_board()
    
    board = np.zeros((u.bline, u.bline), dtype=int)
    running = True
    gameover = False
    
    while running:
       
        for event in pg.event.get():
            
            if event.type == pg.QUIT:
                running = False
            
        if not gameover:
            [row, col] = ai_move1(board, 1)            # First group is assigned to be black
            print("black", row, col)
            if board[row, col] == 0: 
                board[row, col] = 1
                u.draw_stone(surface, [row, col], 1)
            gameover = u.check_winner(board)
            
        if not gameover:
            [row, col] = ai_move2(board, -1)           # Second group is assigned to be white
            print("white", row, col)
            if board[row, col] == 0: 
                board[row, col] = -1
                u.draw_stone(surface, [row, col], -1)
            gameover = u.check_winner(board)
                
        if gameover:
            u.print_winner(surface, gameover)
                
        
    pg.quit()
    
    
if __name__ == '__main__':
    battle(gp1.ai_move, u.random_move)
    battle(u.random_move, gp1.ai_move)
