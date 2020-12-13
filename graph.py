import maxflow
import cv2
import numpy as np

from img_io import read_img, show_img, write_img

INF = 1.0e8
LARGE = 1.0e12

class Graph():
    def __init__(self, h, w):
        self.consider_old_seams = True
        self.grad_energy = True
        self.w, self.h = w, h
        self.filled = np.zeros((self.h, self.w), np.int32)
        self.canvas = np.zeros((self.h, self.w, 3), np.int32)
        self.graph = maxflow.Graph[float]()
        self.node_ids = self.graph.add_grid_nodes((self.h, self.w))
        # print(self.node_ids)
        self.vertical_seams = np.zeros((self.h-1, self.w, 13))
        self.horizontal_seams = np.zeros((self.h, self.w-1, 13))
        self.best_opt = []

    def init_graph(self, new_patch):
        new_h, new_w = new_patch.shape[:2]
        self.filled[:new_h, :new_w] = 1
        self.canvas[:new_h, :new_w] = new_patch
        self.best_opt.append((0, 0))

    def fast_cost_fn(self, new_patch, row_range, col_range):
        new_value = new_patch
        new_h, new_w = new_patch.shape[:2]
        mask = np.expand_dims(self.filled, -1)
        canvas_with_mask = mask*self.canvas

        # summed table for speed up
        summed_table = np.zeros((self.h+1, self.w+1), np.float64)
        summed_table[1:, 1:] = np.power(canvas_with_mask, 2).sum(2)
        summed_table = summed_table.cumsum(axis=0).cumsum(axis=1)
        y_start, x_start = new_h, new_w
        y, x = len(row_range), len(col_range)
        term2 = summed_table[0:y, 0:x] + \
            summed_table[y_start:y_start+y, x_start:x_start+x] - \
            summed_table[y_start:y_start+y, 0:x] - \
            summed_table[0:y, x_start:x_start+x]

        # FFT for speed up
        term3 = cv2.filter2D(canvas_with_mask, -1, new_value, anchor=(0, 0))[0:y, 0:x].sum(axis=2)
        term1 = cv2.filter2D(np.tile(mask, (1, 1, 3)), -1, np.power(new_value, 2), anchor=(0, 0))[0:y, 0:x].sum(axis=2)
 
        # summed table for mask count calculation speed up
        summed_table_mask = np.zeros((self.h+1, self.w+1), np.float64)
        summed_table_mask[1:, 1:] = mask[:, :, 0]
        summed_table_mask = summed_table_mask.cumsum(axis=0).cumsum(axis=1)
        mask_count = summed_table_mask[0:y, 0:x] + \
            summed_table_mask[y_start:y_start+y, x_start:x_start+x] - \
            summed_table_mask[y_start:y_start+y, 0:x] - \
            summed_table_mask[0:y, x_start:x_start+x]
        
        # must cover some area
        cost_table = term1.astype(np.float64)+term2-2*term3
        cost_table = cost_table.astype(np.float64)
        zero_mask = mask_count == 0
        not_zero_mask = np.logical_not(zero_mask)
        # low_overlap_mask = mask_count < int((new_h*new_w)*0.2)
        # high_overlap_mask = mask_count > int((new_h*new_w)*0.5)
        # extreme_overlap_mask = np.logical_or(low_overlap_mask, high_overlap_mask)
        # extreme_overlap_mask = np.logical_and(extreme_overlap_mask, not_zero_mask)
        # normal_overlap_mask = np.logical_and(np.logical_not(extreme_overlap_mask), not_zero_mask).astype(np.float64)
        # cost_table = INF*zero_mask+(cost_table/(mask_count+1e-8)) \
        #     *normal_overlap_mask+mask_count.astype(np.float64)*LARGE*extreme_overlap_mask

        cost_table = INF*zero_mask+(cost_table/(mask_count+1e-8))*not_zero_mask
        return cost_table, mask_count


    def cost_fn(self, new_patch):
        new_t, new_l, new_h, new_w, new_value = new_patch
        new_r, new_b = min(new_l + new_w, self.w), min(new_t + new_h, self.h)
        overlap_area = self.filled[new_t:new_b, new_l:new_r] 
        overlap_count = overlap_area.sum()
        canvas_with_mask = np.expand_dims(overlap_area, -1)*(self.canvas[new_t:new_b, new_l:new_r]-new_value)
        overlap_cost = np.power(canvas_with_mask, 2).astype(np.float64).sum()
        if overlap_count == 0:
            return INF, 0
        if overlap_count > int((new_h*new_w)*0.9) or overlap_count < int((new_h*new_w)*0.2):
            return overlap_count * LARGE, overlap_count
        overlap_cost /= overlap_count 
        return overlap_cost, overlap_count

    def weight_fn(self, new_value, row_idx, col_idx, new_t, new_l, vertical, 
                  ord=2, eps=1e-8, old_value_1=None, old_value_2=None):
        # could specify old_value
        if old_value_1 is None:
            old_value_1 = self.canvas[row_idx][col_idx]
        if old_value_2 is None:
            old_value_2 = self.canvas[row_idx+1][col_idx] if vertical \
                else self.canvas[row_idx][col_idx+1]
        ws = np.linalg.norm(old_value_1-new_value[row_idx-new_t][col_idx-new_l], ord=ord)
        if vertical:
            wt = np.linalg.norm(
                old_value_2-new_value[row_idx-new_t+1][col_idx-new_l], ord=ord)
            grad_s = np.linalg.norm(old_value_1-old_value_2, ord=ord)
            grad_t = np.linalg.norm(
                new_value[row_idx-new_t][col_idx-new_l]-new_value[row_idx-new_t+1][col_idx-new_l], ord=ord)
        else:
            wt = np.linalg.norm(
                old_value_2-new_value[row_idx-new_t][col_idx-new_l+1], ord=ord)
            grad_s = np.linalg.norm(old_value_1-old_value_2, ord=ord)
            grad_t = np.linalg.norm(
                new_value[row_idx-new_t][col_idx-new_l]-new_value[row_idx-new_t][col_idx-new_l+1], ord=ord)
        w = ws+wt
        # integrate grad into energy function
        if self.grad_energy:
            w /= (grad_s+grad_t)*2+eps
        return w
        
    def create_graph(self, old_patch, new_patch):
        new_t, new_l, new_h, new_w, new_value = new_patch
        new_r, new_b = new_l + new_w, new_t + new_h
        src_tedge_count = 0
        sink_tedge_count = 0
        nodes = []
        edges = []
        tedges = []
        node_count = self.h*self.w
        for row_idx in range(new_t, new_b):
            for col_idx in range(new_l, new_r):
                # only consider filled pixels
                if not self.filled[row_idx, col_idx]:
                    continue
                # nodes.append((row_idx, col_idx))
                if row_idx < new_b - 1 and self.filled[row_idx+1, col_idx]:
                    # add old seam nodes 
                    if self.consider_old_seams and self.vertical_seams[row_idx, col_idx][0] > 0:
                        nodes.append(node_count)
                        weight = self.vertical_seams[row_idx, col_idx, 0]
                        tedges.append((node_count, 0, weight))
                        weight = self.weight_fn(
                            new_value, row_idx, col_idx, new_t, new_l, True, 
                            old_value_1=self.vertical_seams[row_idx, col_idx, 1:4], 
                            old_value_2=self.vertical_seams[row_idx, col_idx, 4:7])
                        edges.append((self.node_ids[row_idx][col_idx],
                                      node_count, weight))
                        weight = self.weight_fn(
                            new_value, row_idx, col_idx, new_t, new_l, True,
                            old_value_1=self.vertical_seams[row_idx, col_idx, 7:10],
                            old_value_2=self.vertical_seams[row_idx, col_idx, 10:13])
                        edges.append((self.node_ids[row_idx+1][col_idx],
                                      node_count, weight))
                        node_count += 1
                    else:
                        weight = self.weight_fn(new_value, row_idx, 
                                                col_idx, new_t, new_l, True)
                        edges.append((self.node_ids[row_idx][col_idx],
                                      self.node_ids[row_idx+1][col_idx],
                                      weight))
                if col_idx < new_r - 1 and self.filled[row_idx, col_idx+1]:
                    if self.consider_old_seams and self.horizontal_seams[row_idx, col_idx][0] > 0:
                        nodes.append(node_count)
                        weight = self.horizontal_seams[row_idx, col_idx, 0]
                        tedges.append((node_count, 0, weight))
                        weight = self.weight_fn(
                            new_value, row_idx, col_idx, new_t, new_l, False,
                            old_value_1=self.horizontal_seams[row_idx, col_idx, 1:4],
                            old_value_2=self.horizontal_seams[row_idx, col_idx, 4:7])
                        edges.append((self.node_ids[row_idx][col_idx],
                                      node_count, weight))
                        weight = self.weight_fn(
                            new_value, row_idx, col_idx, new_t, new_l, False,
                            old_value_1=self.horizontal_seams[row_idx, col_idx, 7:10],
                            old_value_2=self.horizontal_seams[row_idx, col_idx, 10:13])
                        edges.append((self.node_ids[row_idx][col_idx+1],
                                      node_count, weight))
                        node_count += 1
                    else:
                        weight = self.weight_fn(new_value, row_idx, 
                                                col_idx, new_t, new_l, False)
                        edges.append((self.node_ids[row_idx][col_idx], 
                                      self.node_ids[row_idx][col_idx+1],
                                      weight))
                if row_idx > new_t and not self.filled[row_idx-1, col_idx] or \
                        row_idx < new_b-1 and not self.filled[row_idx+1, col_idx] or \
                        col_idx > new_l and not self.filled[row_idx, col_idx-1] or \
                        col_idx < new_r-1 and not self.filled[row_idx, col_idx+1]:
                    tedges.append((self.node_ids[row_idx][col_idx], 0, np.inf))
                    src_tedge_count += 1
                if row_idx == new_t and row_idx > 0 and self.filled[row_idx-1, col_idx] or \
                        row_idx == new_b-1 and row_idx < self.h-1 and self.filled[row_idx+1, col_idx] or \
                        col_idx == new_l and col_idx > 0 and self.filled[row_idx, col_idx-1] or \
                        col_idx == new_r-1 and col_idx < self.w-1 and self.filled[row_idx, col_idx+1]:
                    # if src_tedge_count == sink_tedge_count // 2:
                    #     continue
                    tedges.append((self.node_ids[row_idx][col_idx], np.inf, 0))
                    sink_tedge_count += 1
                
        # if src_tedge_count == 0:
        #     tedges.append((self.node_ids[(new_t+new_b)//2][(new_l+new_r)//2], 0, np.inf)) 
        return nodes, edges, tedges

    def blend(self, pattern, row=-1, col=-1, mode='random', k=10, new_pattern_size=None):
        if mode == 'opt_sub':
            h, w = pattern.shape[:2]
            if new_pattern_size is None:
                new_pattern_size = (pattern.shape[0]//2, pattern.shape[1]//2)
            row_rand = np.random.randint(0, h-new_pattern_size[0]+1)
            col_rand = np.random.randint(0, w-new_pattern_size[1]+1)
            pattern = pattern[row_rand:row_rand+new_pattern_size[0],
                              col_rand:col_rand+new_pattern_size[1]]
        h, w = pattern.shape[:2]
        print(h, w)
        max_overlap = max(int(h*w*0.8), h*w-self.h*self.w+self.filled.sum())
        min_overlap = int(h*w*0.1)
        min_cost = np.inf
        if row == -1 or col == -1:
            if mode == 'random':
                row = np.random.randint(0, self.h-h+1)
                col = np.random.randint(0, self.w-w+1)
            elif mode == 'opt_whole' or mode == 'opt_sub':
                row_range = list(range(0, self.h-h+1, 1)) if row == -1 else [row]
                col_range = list(range(0, self.w-w+1, 1)
                                 ) if col == -1 else [col]
                cost_table, mask_count = self.fast_cost_fn(pattern, row_range, col_range)
                # min_idx = cost_table.reshape((-1)).argmin()
                cost_table_flatten = cost_table.reshape((-1))
                mask_count_flatten = mask_count.reshape((-1))
                valid_mask = (mask_count_flatten <= max_overlap) * \
                    (mask_count_flatten >= min_overlap)
                if valid_mask.sum() == 0:
                    p_table_flatten = (mask_count_flatten == mask_count_flatten.min()).astype(np.float32)
                else:
                    sigma = np.std(pattern.reshape(-1, 3), axis=0)
                    sigma_sqr = (sigma*sigma).sum()
                    p_table_flatten = -cost_table_flatten*k/sigma_sqr
                    p_table_flatten = np.exp(p_table_flatten)
                    p_table_flatten = p_table_flatten*valid_mask
                p_table_flatten /= p_table_flatten.sum()
                p_table_flatten = np.cumsum(p_table_flatten)
                rand_num = np.random.rand()
                min_idx = 0
                for i, p in enumerate(p_table_flatten):
                    if rand_num < p:
                        min_idx = i
                        break
                row = min_idx // len(col_range)
                col = min_idx % len(col_range)
                print(row, col, cost_table[0][0], mask_count[0, 0])
                self.best_opt.append((row, col))
            else:
                raise NotImplementedError()
                # for row_idx in range(0, self.h-h, 10):
                #     for col_idx in range(0, self.w-w, 10):
                #         cost = self.cost_fn((row_idx, col_idx, h, w, pattern))
                #         if cost == 0:
                #             continue
                #         if cost < min_cost:
                #             min_cost = cost
                #             min_row = row_idx
                #             min_col = col_idx
        nodes, edges, tedges = self.create_graph(None, (row, col, h, w, pattern))
        graph = maxflow.Graph[float]()
        # node_list = graph.add_nodes(len(self.node_ids))
        # print(node_list)
        # graph.add_grid_nodes((self.h, self.w))
        final_nodes = graph.add_nodes(len(nodes)+self.h*self.w)
        edge_weights = np.zeros((self.h, self.w, 2))
        for edge in edges:
            graph.add_edge(edge[0], edge[1], edge[2], edge[2])
            row_, col_ = edge[0]//self.w, edge[0]%self.w
            if edge[1] == edge[0]+1: 
                edge_weights[row_, col_, 1] = edge[2]
            else:
                edge_weights[row_, col_, 0] = edge[2]
        for tedge in tedges:
            graph.add_tedge(tedge[0], tedge[1], tedge[2])
        flow = graph.maxflow()
        sgm = graph.get_grid_segments(self.node_ids)
        # print(sgm[row:h, col:w]) 
        # sgm = sgm * self.filled
        assert sgm.sum() == sgm[row:row+h, col:col+w].sum()
        print(sgm.sum()/self.filled[row:row+h, col:col+w].sum())
        # print((sgm*self.filled)[row:row+h:3, col:col+w:3].astype(int))
        # exit(0)
        for row_idx in range(row, row+h):
            for col_idx in range(col, col+w):
                if self.consider_old_seams:
                    if row_idx < row+h-1 and self.filled[row_idx, col_idx] and \
                        self.filled[row_idx+1, col_idx]:
                        if not sgm[row_idx, col_idx] and sgm[row_idx+1, col_idx]:
                            self.vertical_seams[row_idx][col_idx][0] = \
                                edge_weights[row_idx][col_idx][0]
                            self.vertical_seams[row_idx][col_idx][1:] = \
                                np.concatenate([
                                    self.canvas[row_idx, col_idx],
                                    self.canvas[row_idx+1, col_idx],
                                    pattern[row_idx-row, col_idx-col],
                                    pattern[row_idx-row+1, col_idx-col]
                                ], axis=-1)
                        if sgm[row_idx, col_idx] and not sgm[row_idx+1, col_idx]:
                            self.vertical_seams[row_idx][col_idx][0] = \
                                edge_weights[row_idx][col_idx][0]
                            self.vertical_seams[row_idx][col_idx][1:] = \
                                np.concatenate([
                                    pattern[row_idx-row, col_idx-col],
                                    pattern[row_idx-row+1, col_idx-col],
                                    self.canvas[row_idx, col_idx],
                                    self.canvas[row_idx+1, col_idx],
                                ], axis=-1)
                    
                    if col_idx < col+w-1 and self.filled[row_idx, col_idx] and \
                        self.filled[row_idx, col_idx+1]:
                        if not sgm[row_idx, col_idx] and sgm[row_idx, col_idx+1]:
                            self.horizontal_seams[row_idx][col_idx][0] = \
                                edge_weights[row_idx][col_idx][1]
                            self.horizontal_seams[row_idx][col_idx][1:] = \
                                np.concatenate([
                                    self.canvas[row_idx, col_idx],
                                    self.canvas[row_idx, col_idx+1],
                                    pattern[row_idx-row, col_idx-col],
                                    pattern[row_idx-row, col_idx-col+1]
                                ], axis=-1)
                        if sgm[row_idx, col_idx] and not sgm[row_idx, col_idx+1]:
                            self.horizontal_seams[row_idx][col_idx][0] = \
                                edge_weights[row_idx][col_idx][1]
                            self.horizontal_seams[row_idx][col_idx][1:] = \
                                np.concatenate([
                                    pattern[row_idx-row, col_idx-col],
                                    pattern[row_idx-row, col_idx-col+1],
                                    self.canvas[row_idx, col_idx],
                                    self.canvas[row_idx, col_idx+1]
                                ], axis=-1)
                    
                if not self.filled[row_idx, col_idx] or self.filled[row_idx, col_idx] and sgm[row_idx, col_idx]:
                        self.canvas[row_idx, col_idx] = pattern[row_idx-row, col_idx-col]
        self.filled[row:row+h, col:col+w] = 1
        # print(self.filled.sum())
        # self.show_canvas()

    
    

    def show_canvas(self):
        show_img(self.canvas)
    
    def write_canvas(self, fn):
        write_img(self.canvas, fn)

if __name__ == '__main__':
    
    # g = Graph(10, 10)
    # g.init_graph(np.ones((5, 5, 3), np.int32)*2) 
    # nodes, edges, tedges = g.create_graph(
    #     None, (2, 2, 5, 5, np.zeros((5, 5, 3)).astype(np.int32)))
    # graph = maxflow.Graph[float]()
    # # node_list = graph.add_nodes(len(self.node_ids)) 
    # # print(node_list)
    # nodes = graph.add_grid_nodes((g.h, g.w))
    # for edge in edges:
    #     x = np.random.randint(2, 10)
    #     graph.add_edge(edge[0], edge[1], x, x)
    # for tedge in tedges:
    #     print(tedge)
    #     graph.add_tedge(tedge[0], tedge[1], tedge[2])
    # print(graph.get_grid_segments(nodes))
    
    # plot_graph_2d(graph, (10, 10)) 
    # # g.blend(np.ones((5, 5, 3), np.int32)* 1, mode='opt_whole')
    # exit(0)  

    
    path = 'C:\\Users\\13731\\Dropbox\\My PC (LAPTOP-VJ2F61DB)\\Desktop\\green.gif'
    pattern = read_img(path)
    h, w = pattern.shape[:2]
    pattern = pattern.astype(np.int32)
    # print(pattern.min(), pattern.max())
    g = Graph(int(3*h), int(3*w))
    g.init_graph(pattern)
    print(pattern.shape)
    print(g.canvas.shape)
    # cv2.imshow("image", g.canvas.asty pe(np.uint8))
    # cv2.waitKey() 

    # g.blend(pattern, mode='random', min_row=0, min_col=0) 
    # g.blend(pattern, mode='random', min_row=0, min_col=w//2)
    # # g.blend(pattern, mode='random', min_row=0, min_col=w)
    # # g.blend(pattern, mode='random', min_row=0, min_col=0)
    # g.blend(pattern, mode='random', min_row=h//2, min_col=0) 
    # g.blend(pattern, mode='random', min_row=h, min_col=0)

    # g.blend(pattern, row=0, col=144)
    # g.blend(pattern, row=96, col=0)
    # # g.blend(pattern, row=96, col=144)
    # g.blend(pattern, row=64, col=96)
    # # g.blend(pattern, row=0, col=48)
    # g.show_canvas()
    # exit(0) 


    while g.filled.sum() < g.h * g.w:
        g.blend(pattern, mode='opt_whole') 
        # g.blend(pattern, mode='opt_sub', new_pattern_size=(100, 100))
        # break
    # for i in range(7):
    #     print('after')   
    #     g.blend(pattern, mode='opt_whole')
    g.show_canvas()
    exit(0)
    for i in range(3): 
        # g.blend(pattern, int(h*0.5), 0) 
        g.blend(pattern, mode='opt_whole', min_row=0) 
    g.blend(pattern, mode='opt_whole', min_col=0)
    for i in range(3):
        # g.blend(pattern, int(h*0.5), 0)
        g.blend(pattern, mode='opt_whole', min_row=int(0.5*h))
        # g.blend(pattern, mode='opt_whole', min_row=0)
        # g.blend(pattern, mode='opt_whole', min_row=0)
        # g.blend(pattern, mode='opt_whole')
        # g.blend(pattern, w//2, w//2)
        # g.blend(pattern, w//2, 0)
    # g.blend(pattern, 0, int(1*w))
    # g.blend(pattern, int(1*h), 0)
    # g.blend(pattern, int(1*h), int(1*w))
    # g.blend(pattern, int(0.25*w), 0)
    # g.blend(pattern, 0, int(0.25*w))
    # g.blend(pattern, int(0.25*h), int(0.25*w))
    exit(0)
