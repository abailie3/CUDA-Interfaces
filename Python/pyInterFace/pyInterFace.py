#from ctypes import *
#print('TEST')
#def multiply(a,b):
#    print("MULTIPLYYY")
#    c = 0
#    for i in range(0,a):
#        c = c + b
#    return c

#def returnStructArray(row, col):
#    print("Return array, rows: " + str(row) + " cols: " + str(col))
#    out = []
#    for r in range(0, row-1):
#        for c in range(0, col-1):
#            out.append((float)(r+c))
#    s_arr = StructArray(out, row, col)
#    return byref(s_arr)
#    #return byref((c_float * (rows * cols))(out))

#class StructArray(Structure):
#    def __init__(self, list, rows, cols):
#        _fields_ = [('rows', c_int),
#                    ('cols', c_int),
#                    ('data', c_float * (rows * cols))]
#        self.rows = (c_int)(rows)
#        self.cols = (c_int)(cols)
#        print(rows)
#        print(cols)
#        c_data = (c_float * (rows * cols))(*list)
#        self.data = cast(c_data, POINTER(c_float * (rows * cols)))
#        #self.data = (c_float * (rows * cols))(*list)
