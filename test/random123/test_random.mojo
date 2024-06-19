import random123
from python import Python
from math import sqrt
from testing import assert_equal, assert_almost_equal, assert_true
from utils.numerics import isnan, isinf


fn test_uniform() raises:
    alias dtypes = List(DType.bfloat16, DType.float16, DType.float32, DType.float64)
    @parameter
    for i in range(len(dtypes)):
        var key = random123.key(12343)
        var samples = random123.uniform[dtypes[i]](key^, 10000000, -1, 2)

        assert_equal(len(samples), 10000000)

        for s in samples:
            assert_true(s[] >= -1 and s[] < 2)
            assert_true(not isnan(s[]) and not isinf(s[]))


fn test_normal() raises:
    alias dtypes = List(DType.bfloat16, DType.float16, DType.float32, DType.float64)
    @parameter 
    for n in range(len(dtypes)): 
        alias dtype = dtypes[n]
        var stats = Python.import_module("scipy.stats")
        var np = Python.import_module("numpy")

        var key1 = random123.key(123)
        var key2 = random123.key(123)
        var samples1 = random123.normal[dtype](key1^, 100000, 2, 2)
        var samples2 = random123.normal[dtype](key2^, 100000, 2, 2)

        for i in range(len(samples1)):
            assert_equal(samples1[i], samples2[i])

        var key3 = random123.key(123123)
        var mean = 1.0
        var std = 2.0

        var samples3 = random123.normal[dtype](key3^, 1000000, mean, std)
        
        var data = np.zeros(len(samples3), dtype=np.float32)
        for i in range(len(samples3)):
            assert_true(not isinf(samples3[i]) and not isnan(samples3[i]))
            data[i] = samples3[i]

        var p_value = stats.kstest(data, 'norm', (mean, std))[1]
        
        @parameter
        if not dtype.is_half_float():
            assert_true(p_value > 0.05)
            
def main():
    test_uniform()
    test_normal()
