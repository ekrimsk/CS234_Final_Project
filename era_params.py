import numpy as np 

# Super class with parameters that all ERA subclasses inherit 
class era_params:
    def __init__(self, n):
        self.lims_low = -0.0393 * np.ones(n)
        self.lims_high = 0.0743 * np.ones(n)
        self.r = 0.05 * np.ones(n)              # wrap radii
        

class era_1_params(era_params):
    def __init__(self):
        n = 1
        era_params.__init__(self, n)
        self.k = 1000
        self.b = 500

class era_2_params(era_params):
    def __init__(self):
        n = 2
        era_params.__init__(self, n)
        self.k = np.array([770.9,  1387.6])
        self.b = np.array([183.9,  331.1])


class era_3_params(era_params):
    def __init__(self):
        n = 3
        era_params.__init__(self, n)
        self.k = np.array([357.4,  643.2,  1157.8])
        self.b = np.array([85.3,  153.5,  276.3])

class era_4_params(era_params):
    def __init__(self):
        n = 4
        era_params.__init__(self, n)
        self.k = np.array([109.4,  196.8,  354.3,  637.8,  1148.0])
        self.b = np.array([43.4,  78.1,  140.6,  253.0])

class era_5_params(era_params):
    def __init__(self):
        n = 5
        era_params.__init__(self, n)
        self.k = np.array([96.5,  173.7,  312.6,  562.7,  1012.9])
        self.b = np.array([26.1,  47.0,  84.5,  152.2,  273.9])

class era_6_params(era_params):
    def __init__(self):
        n = 6
        era_params.__init__(self, n)
        self.k = np.array([52.3,  94.2,  169.5,  305.1,  549.1,  988.4])
        self.b = np.array([ 12.5,  22.5,  40.4,  72.8,  131.0,  235.8])


class era_7_params(era_params):
    def __init__(self):
        era_params.__init__(self, 7)
        self.k = np.array([28.7,  51.6,  92.9,  167.2,  301.0,  541.8,  975.2])
        self.b = np.array([6.8,  12.3,  22.2,  39.9,  71.8,  129.3,  232.7])


class era_8_params(era_params):
    def __init__(self):
        era_params.__init__(self, 8)
        self.k = np.array([17.9,  32.3,  58.1,  104.5,  188.1,  338.6,  609.5,  1097.2])
        self.b = np.array([4.3,  7.7,  13.9,  24.9,  44.9,  80.8,  145.4,  261.8])



class era_10_params(era_params):
    def __init__(self):
        era_params.__init__(self, 10)
        self.k = np.array([12.8,  20.6,  33.2,  53.4,  86.1,  138.8,  223.7,  360.5,  581.0,  936.3])
        self.b = np.array([3.0,  4.9,  7.9,  12.8,  20.5,  33.1,  53.4,  86.0,  138.6,  223.4])



class era_12_params(era_params):
    def __init__(self):
        era_params.__init__(self, 12)
        self.k = np.array([13.5,  19.4,  28.0,  40.5,  58.4,  84.2,  121.5,  175.4,  253.1,  365.1,  526.9,  760.2])
        self.b = np.array([3.2,  4.6,  6.7,  9.7,  13.9,  20.1,  29.0,  41.8,  60.4,  87.1,  125.7,  181.4])



class era_16_params(era_params):
    def __init__(self):
        era_params.__init__(self, 16)
        self.k = np.array([13.6,  17.4,  22.2,  28.4,  36.3,  46.5,  59.4,  76.0,  97.2,  124.3,  159.0,  203.4,  260.1,  332.7,  425.5,  544.2])
        self.b = np.array([3.2,  4.1,  5.3,  6.8,  8.7,  11.1,  14.2,  18.1,  23.2,  29.7,  37.9,  48.5,  62.1,  79.4,  101.5,  129.9])




def get_era_params(n):
    if (n == 1):
        return era_1_params()
    elif (n == 2):
        return era_2_params()
    elif (n == 3):
        return era_3_params()
    elif (n == 4):
        return era_4_params()
    elif (n == 5):
        return era_5_params()
    elif (n == 6):
        return era_6_params()
    elif (n == 7):
        return era_7_params()
    elif (n == 8):
        return era_8_params()
    elif (n == 10):
        return era_10_params()
    elif (n == 12):
        return era_12_params()
    elif (n == 16):
        return era_16_params()
    else:
        RaiseException('Invalid number of clutches')
