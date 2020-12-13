import sys
import time
import numpy as np

from graph import Graph
from img_io import read_img
DEFAULT_PATH_1 = 'data\\green.gif'
DEFAULT_PATH_2 = 'data\\strawberries2.gif'
DEFAULT_PATH_3 = 'data\\akeyboard_small.gif'

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH_2
    pattern = read_img(path).astype(np.int32)
    h, w = pattern.shape[:2]

    target_h = int(sys.argv[2]) if len(sys.argv) > 2 else int(1.5*h)
    target_w = int(sys.argv[3]) if len(sys.argv) > 3 else int(1.5*w)
    mode = int(sys.argv[4]) if len(sys.argv) > 4 else 4 

    g = Graph(target_h, target_w)
    
    if mode == 1:
        g.init_graph(pattern)
        while g.filled.sum() < g.h * g.w:
            g.blend(g.match_patch(pattern, mode='opt_sub', k=1, new_pattern_size=(h//2, w//2)))
        g.show_canvas()
        g.write_canvas('res\\test1.jpg')
        # for i in range(10):
        #     g.blend(g.match_patch(pattern, mode='random'))
        #     g.show_canvas() 

    elif mode == 2:
        g.init_graph(pattern[:h//2])
        start_row = 0
        while g.filled.sum() < g.w*g.h:
            while g.filled.sum() < (start_row+(h//2))*g.w:
                g.blend(g.match_patch(pattern, mode='opt_sub', k=100, 
                                      row=start_row, new_pattern_size=(h//2, w//2)))
                # g.show_canvas()
                print(g.filled.sum(), (start_row+(h//2))*g.w)
            if g.h-(h//2)-start_row < h//4:
                start_row = g.h-(h//2)
            else:
                pattern_info = g.match_patch(
                    pattern, mode='opt_sub', k=100, new_pattern_size=(h//2, w//2))
                g.blend(pattern_info)
                # g.show_canvas() 
                start_row = pattern_info[0]
                print(start_row)
        # g.show_canvas()
        g.write_canvas('res\\test1.jpg')
            # g.show_canvas()
            # new_h = pattern_info[0]+pattern_info[2]
            # while g.filled.sum() < new_h * g.w:
            #     g.blend(g.match_patch(pattern, mode='opt_whole', row=pattern_info[0]))
            #     g.show_canvas()
            #     print(g.filled.sum()) 
        # g.show_canvas()
    elif mode == 3:
        g.init_graph(pattern)
        start_row = 0
        while g.filled.sum() < g.w*g.h:
            while g.filled.sum() < (start_row+h)*g.w:
                g.blend(g.match_patch(pattern, mode='opt_whole', k=100,
                                      row=start_row))
                # g.show_canvas()
            if g.h-h-start_row < h//2:
                start_row = g.h-h
            else:
                pattern_info = g.match_patch(
                    pattern, mode='opt_whole', k=100)
                g.blend(pattern_info)
                # g.show_canvas()
                start_row = pattern_info[0]
                print(start_row)
        # g.show_canvas()
        g.write_canvas('res\\test1.jpg')
        # g.show_canvas()
        # new_h = pattern_info[0]+pattern_info[2]
        # while g.filled.sum() < new_h * g.w:
        #     g.blend(g.match_patch(pattern, mode='opt_whole', row=pattern_info[0]))
        #     g.show_canvas()
        #     print(g.filled.sum())
        # g.show_canvas()
    elif mode == 4:
        g.init_graph(pattern)
        for i in range(1):
            g.match_patch(pattern, mode='opt_whole')
        # g.blend(g.match_patch(pattern, row=96, col=0))
        # g.blend(g.match_patch(pattern, row=0, col=96))
        # g.blend(g.match_patch(pattern, row=96, col=96))
        # g.show_canvas()
        # g.write_canvas('res\\test1.jpg')
        # g.blend(g.match_patch(pattern, row=64, col=64))
        # g.show_canvas()
        # g.write_canvas('res\\test2.jpg')

        # g.blend(pattern, mode='opt_sub', new_pattern_size=(100, 100))
        # break
    # for i in range(7):
    #     print('after')
    #     g.blend(pattern, mode='opt_whole')
    

if __name__ == '__main__':
    start = time.time()
    main()
    print('time', time.time()-start)

