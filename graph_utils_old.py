import maxflow
import cv2

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

INF = 1.0e12
LARGE = 1.0e12

class Graph():
    def __init__(self, h, w):
        self.w, self.h = w, h
        self.filled = np.zeros((self.h, self.w), np.int32)
        self.canvas = np.zeros((self.h, self.w, 3), np.int32)
        self.graph = maxflow.Graph[float]()
        self.node_ids = self.graph.add_grid_nodes((self.h, self.w))
        # print(self.node_ids)
        self.best_opt = []

    def init_graph(self, new_patch):
        new_h, new_w = new_patch.shape[:2]
        self.filled[:new_h, :new_w] = 1
        self.canvas[:new_h, :new_w] = new_patch
        self.best_opt.append((0, 0))

    def get_weight_matrix(self, new_patch, new_l, new_t):
        # new_h, new_w = new_patch.shape[:2]
        # new_r, new_b = new_l + new_w, new_t + new_h 
        # for row_idx in range(new_t, new_b):
        #     for col_idx in range(new_l, new_r):
        #         if not self.filled[row_idx, col_idx]:
        #             continue
        new_patch_l_shift = np.roll(new_patch, 1, axis=1)
        new_patch_t_shift = np.roll(new_patch, 1, axis=0)

    def fast_cost_fn(self, new_patch, row_range, col_range):
        new_value = new_patch
        new_h, new_w = new_patch.shape[:2]
        # print(new_h, new_w)
        mask = np.expand_dims(self.filled, -1)
        canvas_with_mask = mask*self.canvas
        # print(canvas_with_mask, row_range)
        # term1 = np.power(new_value.astype(np.float64), 2).sum() # Xp^2
        # print(new_value*new_value)
        # term2 = np.power(canvas_with_mask, 2) 
        summed_table = np.zeros((self.h+1, self.w+1), np.float64)
        summed_table[1:, 1:] = np.power(canvas_with_mask, 2).sum(2)
        summed_table = summed_table.cumsum(axis=0).cumsum(axis=1)
        # print(summed_table)
        y_start, x_start = new_h, new_w#row_range[-1], col_range[-1]
        y, x = len(row_range), len(col_range)#self.h-y_start+1, self.w-x_start+1
        # print(y, x, y_start, x_start)
        term2 = summed_table[0:y, 0:x] + \
            summed_table[y_start:y_start+y, x_start:x_start+x] - \
            summed_table[y_start:y_start+y, 0:x] - \
            summed_table[0:y, x_start:x_start+x]

        
        # summed_table = np.zeros((self.h+1, self.w+1))
        # summed_table[1:, 1:] = self.filled
        # summed_table = summed_table.cumsum(axis=0).cumsum(axis=1)
        # mask = summed_table[0:y, 0:x] + \
        #     summed_table[y_start:y_start+y, x_start:x_start+x] - \
        #     summed_table[y_start:y_start+y, 0:x] - \
        #     summed_table[0:y, x_start:x_start+x]

        # print(canvas_with_mask.dtype, new_value.dtype)
        term3 = cv2.filter2D(canvas_with_mask, -1, new_value, anchor=(0, 0))[0:y, 0:x].sum(axis=2)
        # print(mask.shape, new_value.shape)
        term1 = cv2.filter2D(np.tile(mask, (1, 1, 3)), -1, np.power(new_value, 2), anchor=(0, 0))[0:y, 0:x].sum(axis=2)

        # print(term3)
        # print(term1.shape)
        # term1 = np.zeros((len(row_range), len(col_range)), np.float64)
        # term3 = np.zeros((len(row_range), len(col_range)), np.float64)
        # term2 = np.zeros((len(row_range), len(col_range)), np.float64)
        # print(term3.shape, row_range, col_range)
        # for row_idx in row_range:
            # for col_idx in col_range:
                # pass
                # term1[row_idx][col_idx] = np.sum(
                    # mask[row_idx:row_idx+new_h, col_idx:col_idx+new_w]*np.power(new_value, 2))
                # term2[row_idx][col_idx] = np.power(
                #     canvas_with_mask[row_idx:row_idx+new_h, col_idx:col_idx+new_w], 2).sum()
                # term3[row_idx][col_idx] = np.sum(canvas_with_mask[row_idx:row_idx+new_h, col_idx:col_idx+new_w]*new_value)
        # print(term3)
        # exit(0)
        summed_table_mask = np.zeros((self.h+1, self.w+1), np.float64)
        summed_table_mask[1:, 1:] = mask[:, :, 0]
        summed_table_mask = summed_table_mask.cumsum(axis=0).cumsum(axis=1)
        mask_count = summed_table_mask[0:y, 0:x] + \
            summed_table_mask[y_start:y_start+y, x_start:x_start+x] - \
            summed_table_mask[y_start:y_start+y, 0:x] - \
            summed_table_mask[0:y, x_start:x_start+x]
        print(mask_count)

        # print(term1[0, 0], term2[0, 0], term3[0, 0])
        
        cost_table = term1.astype(np.float64)+term2-2*term3
        # print(cost_table[0, 0])
        # exit(0)
        cost_table = cost_table.astype(np.float64)
        zero_mask = mask_count == 0
        not_zero_mask = np.logical_not(zero_mask)
        low_overlap_mask = mask_count < int((new_h*new_w)*0.2)
        high_overlap_mask = mask_count > int((new_h*new_w)*0.5)
        extreme_overlap_mask = np.logical_or(low_overlap_mask, high_overlap_mask)
        extreme_overlap_mask = np.logical_and(extreme_overlap_mask, not_zero_mask)
        normal_overlap_mask = np.logical_and(np.logical_not(extreme_overlap_mask), not_zero_mask).astype(np.float64)
        cost_table = INF*zero_mask+(cost_table/(mask_count+1e-8)) \
            *normal_overlap_mask+mask_count.astype(np.float64)*LARGE*extreme_overlap_mask

        return cost_table, mask_count


    def cost_fn(self, new_patch):
        new_t, new_l, new_h, new_w, new_value = new_patch
        new_r, new_b = min(new_l + new_w, self.w), min(new_t + new_h, self.h)
        overlap_area = self.filled[new_t:new_b, new_l:new_r] 
        overlap_count = overlap_area.sum()
        overlap_cost = np.power(np.expand_dims(overlap_area, -1)*(self.canvas[new_t:new_b, new_l:new_r]-new_value), 2).astype(np.float64).sum()

        if overlap_count == 0:
            return INF, 0
        if overlap_count > int((new_h*new_w)*0.9) or overlap_count < int((new_h*new_w)*0.2):
            return overlap_count * LARGE, overlap_count
        overlap_cost /= overlap_count 
        return overlap_cost, overlap_count

    def weight_fn(self, new_value, row_idx, col_idx, new_t, new_l, vertical):
        ws = np.linalg.norm(self.canvas[row_idx][col_idx]-new_value[row_idx-new_t][col_idx-new_l], ord=2)
        if vertical:
            wt = np.linalg.norm(
                self.canvas[row_idx+1][col_idx]-new_value[row_idx-new_t+1][col_idx-new_l], ord=2)
            grad_s = np.linalg.norm(
                self.canvas[row_idx][col_idx]-self.canvas[row_idx+1][col_idx], ord=2)
            grad_t = np.linalg.norm(
                new_value[row_idx-new_t][col_idx-new_l]-new_value[row_idx-new_t+1][col_idx-new_l], ord=2)
        else:
            wt = np.linalg.norm(
                self.canvas[row_idx][col_idx+1]-new_value[row_idx-new_t][col_idx-new_l+1], ord=2)
            grad_s = np.linalg.norm(
                self.canvas[row_idx][col_idx]-self.canvas[row_idx][col_idx+1], ord=2)
            grad_t = np.linalg.norm(
                new_value[row_idx-new_t][col_idx-new_l]-new_value[row_idx-new_t][col_idx-new_l+1], ord=2)
        w = (ws+wt)/((grad_s+grad_t)*2+1e-8)
        return w
        


    def create_graph(self, old_patch, new_patch):
        # old_l, old_t, old_w, old_h = old_patch
        new_t, new_l, new_h, new_w, new_value = new_patch
        new_r, new_b = new_l + new_w, new_t + new_h
        src_tedge_count = 0
        sink_tedge_count = 0
        nodes = []
        edges = []
        tedges = []
        for row_idx in range(new_t, new_b):
            for col_idx in range(new_l, new_r):
                if not self.filled[row_idx, col_idx]:
                    continue
                nodes.append((row_idx, col_idx))

                if row_idx < new_b - 1 and self.filled[row_idx+1, col_idx]:
                    # print(new_value)
                    # print(self.canvas[row_idx, col_idx] -
                    #       new_value[row_idx, col_idx])
                    weight = self.weight_fn(new_value, row_idx, col_idx, new_t, new_l, True)
                    # weight = np.linalg.norm(self.canvas[row_idx][col_idx]-new_value[row_idx-new_t][col_idx-new_l]) + \
                    #     np.linalg.norm(self.canvas[row_idx+1][col_idx] - new_value[row_idx-new_t+1][col_idx-new_l])
                    # print(weight)
                    edges.append((self.node_ids[row_idx][col_idx],
                                   self.node_ids[row_idx+1][col_idx],
                                   weight))
                if col_idx < new_r - 1 and self.filled[row_idx, col_idx+1]:
                    weight = self.weight_fn(
                        new_value, row_idx, col_idx, new_t, new_l, False)
                    # weight = np.linalg.norm(self.canvas[row_idx][col_idx]-new_value[row_idx-new_t][col_idx-new_l]) + \
                    #     np.linalg.norm(self.canvas[row_idx][col_idx+1] -
                    #              new_value[row_idx-new_t][col_idx-new_l+1])
                    edges.append((self.node_ids[row_idx][col_idx],
                                   self.node_ids[row_idx][col_idx+1],
                                   weight))
                if row_idx > new_t and not self.filled[row_idx-1, col_idx] or \
                        row_idx < new_b-1 and not self.filled[row_idx+1, col_idx] or \
                        col_idx > new_l and not self.filled[row_idx, col_idx-1] or \
                        col_idx < new_r-1 and not self.filled[row_idx, col_idx+1]:
                    tedges.append((self.node_ids[row_idx][col_idx], np.inf, 0))
                    src_tedge_count += 1
        for row_idx in range(new_t, new_b):
            for col_idx in range(new_l, new_r):
                if not self.filled[row_idx, col_idx]:
                    continue
                if row_idx == new_t and row_idx > 0 and self.filled[row_idx-1, col_idx] or \
                        row_idx == new_b-1 and row_idx < self.h-1 and self.filled[row_idx+1, col_idx] or \
                        col_idx == new_l and col_idx > 0 and self.filled[row_idx, col_idx-1] or \
                        col_idx == new_r-1 and col_idx < self.w-1 and self.filled[row_idx, col_idx+1]:
                    # if src_tedge_count == sink_tedge_count // 2:
                    #     continue 
                    tedges.append((self.node_ids[row_idx][col_idx], 0, np.inf))
                    sink_tedge_count += 1
                
                
        if src_tedge_count == 0: 
            tedges.append((self.node_ids[(new_t+new_b)//2][(new_l+new_r)//2], np.inf, 0))
        # if new_l > 0:
        #     for row_idx in range(new_t, new_b):
        #         if self.filled[row_idx, new_l] and self.filled[row_idx, new_l-1]:
        #             self.graph.add_tedge(
        #                 self.node_ids[row_idx][new_l], 0, np.inf)
        # if new_r < self.w-1:
        #     for row_idx in range(new_t, new_b):
        #         if self.filled[row_idx, new_r-1] and self.filled[row_idx, new_r]:
        #             self.graph.add_tedge(
        #                 self.node_ids[row_idx][new_r-1], 0, np.inf)
        # if new_t > 0:
        #     for col_idx in range(new_l, new_r):
        #         if self.filled[new_t, col_idx] and self.filled[new_t-1, col_idx]:
        #             print(
        #                 new_t, col_idx, self.filled[new_t, col_idx], self.filled[new_t-1, col_idx], self.node_ids[new_t][col_idx])
        #             self.graph.add_tedge(
        #                 self.node_ids[new_t][col_idx], 0, np.inf)
        # if new_b < self.h-1:
        #     for col_idx in range(new_l, new_r):
        #         if self.filled[new_b-1, col_idx] and self.filled[new_b, col_idx]:
        #             self.graph.add_tedge(
        #                 self.node_ids[new_b-1][col_idx], 0, np.inf)
        return nodes, edges, tedges

    def blend(self, pattern, min_row=-1, min_col=-1, mode='random'):
        h, w = pattern.shape[:2]
        min_cost = np.inf
        if min_row == -1 or min_col == -1:
            if mode == 'random':
                min_row = np.random.randint(10, self.h-h)
                min_col = np.random.randint(10, self.w-w)
            elif mode == 'opt_whole':
                
                row_range = list(range(0, self.h-h+1, 1)) if min_row == -1 else [min_row]
                col_range = list(range(0, self.w-w+1, 1)) if min_col == -1 else [min_col]
                cost_table, mask_count = self.fast_cost_fn(pattern, row_range, col_range)
                min_idx = cost_table.reshape((-1)).argmin()
                min_row = min_idx // len(col_range)
                min_col = min_idx % len(col_range)
                # print(cost_table)
                
                # for row_idx in row_range:
                #     for col_idx in col_range: 
                #         cost, area = self.cost_fn((row_idx, col_idx, h, w, pattern))
                #         assert abs(
                #             area - mask_count[row_idx, col_idx]) < 1, (area, mask_count[row_idx, col_idx])
                #         assert abs(cost - cost_table[row_idx, col_idx]) < 1, (cost,
                #                                                               cost_table[row_idx, col_idx], area, mask_count[row_idx, col_idx])
                #         if (row_idx, col_idx) in self.best_opt:
                #             continue
                #         if cost < min_cost:
                #             min_cost = cost
                #             min_row = row_idx
                #             min_col = col_idx  
                print(min_row, min_col, cost_table[0][0], mask_count[0,0])
                self.best_opt.append((min_row, min_col))
            elif mode == 'opt_sub':
                for row_idx in range(0, self.h-h, 10):
                    for col_idx in range(0, self.w-w, 10):
                        cost = self.cost_fn((row_idx, col_idx, h, w, pattern))
                        if cost == 0:
                            continue
                        if cost < min_cost:
                            min_cost = cost
                            min_row = row_idx
                            min_col = col_idx
        nodes, edges, tedges = self.create_graph(None, (min_row, min_col, h, w, pattern))
        graph = maxflow.Graph[float]()
        # node_list = graph.add_nodes(len(self.node_ids))
        # print(node_list)
        graph.add_grid_nodes((self.h, self.w))
        for edge in edges:
            graph.add_edge(edge[0], edge[1], edge[2], edge[2])
        for tedge in tedges:
            graph.add_tedge(tedge[0], tedge[1], tedge[2])
        flow = graph.maxflow()
        sgm = graph.get_grid_segments(self.node_ids)
        print(sgm.sum()) 
        for row_idx in range(min_row, min_row+h):
            for col_idx in range(min_col, min_col+w):
                if not self.filled[row_idx, col_idx] or self.filled[row_idx, col_idx] and not sgm[row_idx, col_idx]:
                        self.canvas[row_idx, col_idx] = pattern[row_idx -
                                                     min_row, col_idx-min_col]
        self.filled[min_row:min_row+h, min_col:min_col+w] = 1
        # print(self.filled.sum())
        self.show_canvas()

    
    def plot_graph_2d(self, graph, nodes_shape, plot_weights=False,
                      plot_terminals=True, font_size=1):
        """
        Plot the graph to be used in graph cuts
        :param graph: PyMaxflow graph
        :param nodes_shape: patch shape
        :param plot_weights: if true, edge weights are shown
        :param plot_terminals: if true, the terminal nodes are shown
        :param font_size: text font size
        """
        X, Y = np.mgrid[:nodes_shape[0], :nodes_shape[1]]
        
        aux = np.array([Y.ravel(), X[::-1].ravel()]).T
        positions = {i: v for i, v in enumerate(aux)}
        positions['s'] = (-1, nodes_shape[0] / 2.0 - 0.5)
        positions['t'] = (nodes_shape[1], nodes_shape[0] / 2.0 - 0.5)

        print(positions)
        nxgraph = graph.get_nx_graph()
        print("nxgraph created")
        if not plot_terminals:
            nxgraph.remove_nodes_from(['s', 't'])

        plt.clf()
        nx.draw(nxgraph, pos=positions)
        # for u, v, d in nxgraph.edges(data=True):
        #     print(u, v)

        if plot_weights:
            edge_labels = {}
            for u, v, d in nxgraph.edges(data=True):
                edge_labels[(u, v)] = d['weight']
            nx.draw_networkx_edge_labels(nxgraph,
                                         pos=positions,
                                         edge_labels=edge_labels,
                                         label_pos=0.3,
                                         font_size=font_size)

        plt.axis('equal')
        plt.show()

    def show_canvas(self):
        cv2.imshow("image", self.canvas.astype(np.uint8))
        cv2.waitKey()


import imageio
 
def readImg(im_fn):
    im = cv2.imread(im_fn)
    if im is None :
        print('{} cv2.imread failed'.format(im_fn))
        tmp = imageio.mimread(im_fn)
        if tmp is not None:
            imt = np.array(tmp)
            imt = imt[0]
            im = imt[:,:,[2, 1, 0]]
    return im


# g = Graph(10, 10)
# g.init_graph(np.ones((5, 10, 3), np.int32)*2) 
# nodes, edges, tedges = g.create_graph(
#     None, (2, 2, 5, 5, np.zeros((5, 5, 3)).astype(np.int32)))
# graph = maxflow.Graph[float]()
# # node_list = graph.add_nodes(len(self.node_ids))
# # print(node_list)
# graph.add_grid_nodes((g.h, g.w))
# for edge in edges:
#     graph.add_edge(edge[0], edge[1], edge[2], edge[2])
# for tedge in tedges:
#     graph.add_tedge(tedge[0], tedge[1], tedge[2])

# g.plot_graph_2d(graph, (10, 10)) 
# # g.blend(np.ones((5, 5, 3), np.int32)* 1, mode='opt_whole')
# exit(0) 

 
path = 'C:\\Users\\13731\\Dropbox\\My PC (LAPTOP-VJ2F61DB)\\Desktop\\green.gif'
pattern = readImg(path)
h, w = pattern.shape[:2]
pattern = pattern.astype(np.int32)
# print(pattern.min(), pattern.max())
g = Graph(int(1.5*h), int(1.5*w))
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
# exit(0) 
while g.filled.sum() < g.h * g.w:
    
    g.blend(pattern, mode='opt_whole')
    # break
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


g.create_graph(None, (int(0.5 * h), 0, h, w, pattern))
print(g.graph.get_segment) 

flow = g.graph.maxflow()
sgm = g.graph.get_grid_segments(g.node_ids)
print(sgm.sum())
for row_idx in range(int(0.5 * h), int(0.5 * h)+h):
    for col_idx in range(0, w):
        if g.filled[row_idx, col_idx]: 
            if not sgm[row_idx, col_idx]:
                g.canvas[row_idx, col_idx] = pattern[row_idx-int(0.5 * h), col_idx]
        else:
            g.canvas[row_idx, col_idx] = pattern[row_idx-int(0.5 * h), col_idx]
# for row_idx in range(h, int(0.4 * h)+h):
#     for col_idx in range(0, w):
#         g.canvas[row_idx, col_idx] = pattern[row_idx-int(0.4 * h), col_idx]
# g.canvas[:int(0.4 * h)+h, 0:0+w][sgm] = pattern[sgm]

cv2.imshow("image", g.canvas)
cv2.waitKey()


x = np.array([[1,2], [3, 4]])
print(np.roll(x, 1, axis=0))

