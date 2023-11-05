import numpy as np
board = np.zeros((11,11), dtype=int)
print(board)

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

def check_winner(board):
    array11 = np.concatenate((board[-4:,-4:], board[-4:,:],board[-4:,:4]), axis=1)
    array12 = np.concatenate((board[:,-4:], board,board[:,:4]), axis=1)
    array13 = np.concatenate((board[:4,-4:], board[:4,:],board[:4,:4]), axis=1)

    board1 = np.concatenate((array11, array12, array13), axis=0)  # 19¡Á19 ÅÐ¶¨ÆåÅÌ
    
    n=len(board)
    
    i=row
    j=col
    
    indexlist1=list([i+4,j+k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
    A1=list(board1[m[0],m[1]] for m in indexlist1)
    count = 0
    for num in A1:
        if num == board[i][j]:
            count += 1
            if count == 5:
                return board[i][j]
        else:
            count = 0
    
    indexlist2=list([i+k+4,j+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
    A2=list(board1[m[0],m[1]] for m in indexlist2)
    count = 0
    for num in A2:
        if num == board[i][j]:
            count += 1
            if count == 5:
                return board[i][j]
        else:
            count = 0
    
    indexlist3=list([i+k+4,j+k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
    A3=list(board1[m[0],m[1]] for m in indexlist3)
    count = 0
    for num in A3:
        if num == board[i][j]:
            count += 1
            if count == 5:
                return board[i][j]
        else:
            count = 0
    
    indexlist4=list([i-k+4,j+k+4] for k in (-4,-3,-2,-1,0,1,2,3,4))
    A4=list(board1[m[0],m[1]] for m in indexlist4)
    count = 0
    for num in A4:
        if num == board[i][j]:
            count += 1
            if count == 5:
                return board[i][j]
        else:
            count = 0
    
    for i in range(n):
        for j in range(n):
            if board[i][j] == 0:
                return 0
    return 2



def main_template(player_is_black=True):
    
    global bline
    bline = 11                  # the board size is 11x11 => need to draw 11 lines on the board
    
    pg.init()
    surface = draw_board()
    
    board = np.zeros((bline,bline), dtype=int)
    running = True
    gameover = False
    
    if not player_is_black:
        draw_stone(surface, [5, 5], 1)
    
    global row
    global col
    row=5
    col=5
    
    while running:
        
        if check_winner(board) != 0:
            print_winner(surface, winner=check_winner(board))
       
        for event in pg.event.get():              # A for loop to process all the events initialized by the player
            
            if event.type == pg.QUIT:             # terminate if player closes the game window 
                running = False
                
            if event.type == pg.MOUSEBUTTONDOWN and not gameover:        # detect whether the player is clicking in the window
                
                (x,y) = event.pos                                        # check if the clicked position is on the 11x11 center grid
                if (x > pad+3.75*sep) and (x < w_size-pad-3.75*sep) and (y > pad+3.75*sep) and (y < w_size-pad-3.75*sep):
                    row = round((x-pad)/sep-4)     
                    col = round((y-pad)/sep-4)
                    
                    if board[row, col] == 0:                             # update the board matrix if that position has not been occupied
                        color = 1 if player_is_black else -1
                        board[row, col] = color
                        draw_stone(surface, [row, col], color)
        
        
        if check_winner(board) != 0:
            print_winner(surface, winner=check_winner(board))
        
        
        ####################################################################################################
        ######################## Normally Your edit should be within the while loop ########################
        ####################################################################################################
        
        
    pg.quit()
    

if __name__ == '__main__':
    main_template(True)









