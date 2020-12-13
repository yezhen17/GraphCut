import sys
import numpy as np

from graph import Graph
from img_io import read_img
DEFAULT_PATH = 'data\\green.gif'

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    pattern = read_img(path).astype(np.int32)
    h, w = pattern.shape[:2]

    target_h = sys.argv[2] if len(sys.argv) > 2 else 2*h
    target_w = sys.argv[3] if len(sys.argv) > 3 else 2*w

    g = Graph(target_h, target_w)
    g.init_graph(pattern)

    while g.filled.sum() < g.h * g.w:
        g.blend(pattern, mode='opt_whole')
        # g.blend(pattern, mode='opt_sub', new_pattern_size=(100, 100))
        # break
    # for i in range(7):
    #     print('after')
    #     g.blend(pattern, mode='opt_whole')
    g.show_canvas()
    exit(0)

if __name__ == '__main__':
    main()

