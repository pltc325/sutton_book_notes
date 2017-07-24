import math


class Utils(object):
    @staticmethod
    def state2rowcol(state, row_num, col_num):
        row_num = state / col_num
        col_num = state % col_num
        return math.floor(row_num), col_num

    @staticmethod
    def rowcol2state(row, col, col_num):
        s = int(row * col_num + col)
        return int(row * col_num + col)