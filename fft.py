from cmath import exp
from math import pi
import numpy as np
from numpy.fft import rfft, irfft
from numpy import multiply
import cv2


def fftrealpolymul(arr_a, arr_b):  # fft based real-valued polynomial multiplication

    L = len(arr_a) + len(arr_b) - 1
    a_f = rfft(arr_a, L)
    b_f = rfft(arr_b, L)

    return irfft(multiply(a_f, b_f))


# A simple class to simulate n-th root of unity
# This class is by no means complete and is implemented
# merely for FFT and FPM algorithms


class NthRootOfUnity:
    def __init__(self, n, k=1):
        self.k = k
        self.n = n

    def __pow__(self, other):
        if type(other) is int:
            n = NthRootOfUnity(self.n, self.k * other)
            return n

    def __eq__(self, other):
        if other == 1:
            return abs(self.n) == abs(self.k)

    def __mul__(self, other):
        return exp(2*1j*pi*self.k/self.n)*other

    def __repr__(self):
        return str(self.n) + "-th root of unity to the " + str(self.k)

    @property
    def th(self):
        return abs(self.n // self.k)


# The Fast Fourier Transform Algorithm
#
# Input: A, An array of integers of size n representing a polynomial
#        omega, a root of unity
# Output: [A(omega), A(omega^2), ..., A(omega^(n-1))]
# Complexity: O(n logn)
def FFT(A, omega):
    if omega == 1:
        return [sum(A)]
    o2 = omega**2
    C1 = FFT(A[0::2], o2)
    C2 = FFT(A[1::2], o2)
    C3 = [None]*omega.th
    for i in range(omega.th//2):
        C3[i] = C1[i] + omega**i * C2[i]
        C3[i+omega.th//2] = C1[i] - omega**i * C2[i]
    return C3

# The Fast Polynomial Multiplication Algorithm
#
# Input: A,B, two arrays of integers representing polynomials
#        their length is in O(n)
# Output: Coefficient representation of AB
# Complexity: O(n logn)


def FPM(A, B):
    n = 1 << (len(A)+len(B)-2).bit_length()
    o = NthRootOfUnity(n)
    AT = FFT(A, o)
    BT = FFT(B, o)
    C = [AT[i]*BT[i] for i in range(n)]
    # nm = (len(A)+len(B)-1)
    D = [round((a/n).real) for a in FFT(C, o ** -1)]
    while len(D) > 0 and D[-1] == 0:
        del D[-1]
    return D

def FFT_CONV(A, B):
    A = np.reshape(A, (-1))
    B = np.reshape(B, (-1))[::-1]
    return FPM(A, B)[len(A)//2]


# print(FFT_CONV(np.array([1, 2]), np.array([3, 4])))
# print(fftrealpolymul(np.array([1, 2]), np.array([3, 4])))

def fast_cost_fn(canvas_with_mask, new_patch, row_range, col_range, h, w):
    new_value = new_patch
    new_h, new_w = new_patch.shape[:2]
    term1 = np.power(new_value, 2).sum()  # Xp^2
    # print(new_value*new_value)
    # term2 = np.power(canvas_with_mask, 2)
    cost_matrix = np.zeros((len(row_range), len(col_range)))
    summed_table = np.zeros((h+1, w+1))
    summed_table[1:, 1:] = np.power(canvas_with_mask, 2).sum(2)
    summed_table = summed_table.cumsum(axis=0).cumsum(axis=1)
    print(summed_table)
    y, x = row_range[-1], col_range[-1]
    y_start, x_start = h-y+1, w-x+1
    # print(y, x, y_start, x_start)
    term2 = summed_table[0:y_start, 0:x_start] + \
        summed_table[y:y_start+y, x:x_start+x] - \
        summed_table[y:y_start+y, 0:x_start] - \
        summed_table[0:y_start, x:x_start+x]
    print(term2, term1)
    # print(canvas_with_mask.dtype, new_value.dtype)
    # term3 = cv2.filter2D(canvas_with_mask, -1, new_value)[0:y, 0:x].sum(axis=2)
    term3 = np.zeros((len(row_range), len(col_range)))
    print(term3.shape, row_range, col_range)
    for row_idx in row_range:
        for col_idx in col_range:
            term3[row_idx][col_idx] = np.sum(
                canvas_with_mask[row_idx:row_idx+new_h, col_idx:col_idx+new_w]*new_value)
    # print(term3[0, 0], term1, term2[0, 0])
    print(term3)
    # exit(0)
    return term1+term2-2*term3 

src = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [8, 7, 6, 5],
    [4, 3, 2, 1]
])

kernel = np.array([
    [1, 2],
    [3, 4]
])

print(fast_cost_fn(np.stack([src, src, src], -1), 
                   np.stack([kernel, kernel, kernel], -1), (0, 1, 2), (0, 1, 2), 4, 4))
