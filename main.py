import sys
import time
import numpy as np

from graph import Graph
from img_io import read_img


DEFAULT_PATH_1 = 'data/green.gif'
DEFAULT_PATH_2 = 'data/strawberries2.gif'
DEFAULT_PATH_3 = 'data/akeyboard_small.gif'


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH_2
    pattern = read_img(path).astype(np.int32)
    h, w = pattern.shape[:2]

    target_h = int(sys.argv[2]) if len(sys.argv) > 2 else int(2*h)
    target_w = int(sys.argv[3]) if len(sys.argv) > 3 else int(2*w)
    mode = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    g = Graph(target_h, target_w)
    
    # 全局不断子块最佳匹配
    if mode == 1:
        g.init_graph(pattern)
        while g.filled.sum() < g.h * g.w:
            g.blend(g.match_patch(pattern, mode='opt_sub', k=1, new_pattern_size=(h//2, w//2)))
        g.show_canvas()

    # 逐行子块最佳匹配
    elif mode == 2:
        g.init_graph(pattern[:h//2])
        start_row = 0
        while g.filled.sum() < g.w*g.h:
            while g.filled.sum() < (start_row+(h//2))*g.w:
                g.blend(g.match_patch(pattern, mode='opt_sub', k=100, 
                                      row=start_row, new_pattern_size=(h//2, w//2)))
            if g.h-(h//2)-start_row < h//4:
                start_row = g.h-(h//2)
            else:
                pattern_info = g.match_patch(
                    pattern, mode='opt_sub', k=100, new_pattern_size=(h//2, w//2))
                g.blend(pattern_info) 
                start_row = pattern_info[0]
        g.show_canvas()

    # 逐行最佳匹配
    elif mode == 3:
        g.init_graph(pattern)
        start_row = 0
        while g.filled.sum() < g.w*g.h:
            while g.filled.sum() < (start_row+h)*g.w:
                g.blend(g.match_patch(pattern, mode='opt_whole', k=100,
                                      row=start_row))
            if g.h-h-start_row < h//2:
                start_row = g.h-h
            else:
                pattern_info = g.match_patch(
                    pattern, mode='opt_whole', k=100)
                g.blend(pattern_info)
                start_row = pattern_info[0]
        g.show_canvas()
        
    # test
    elif mode == 4:
        g.init_graph(pattern)
        for i in range(1):
            g.match_patch(pattern, mode='opt_whole')
    

if __name__ == '__main__':
    # start = time.time()
    main()
    # print('time consumed', time.time()-start)

