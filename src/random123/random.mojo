from memory.unsafe import bitcast
from math import iota, sqrt, fma, ceildiv, log
from utils.numerics import FPUtils, nextafter
from .utils import recast, _uint_dtype, above_minus_one
from collections import InlineList
from bit import rotate_bits_left
from algorithm import parallelize
from math.polynomial import polynomial_evaluate


@value
struct PRNGKey(Stringable, Representable):
    var key: SIMD[DType.uint32, 2]

    fn __init__(inout self, key: UInt64):
        self.key = bitcast[DType.uint32, 2](UInt64(key))

    fn __repr__(self) -> String:
        return str("PRNGKey(") + str(self.key) + str(")")

    fn __str__(self) -> String:
        return str(self.key)

    fn __getitem__(self, index: Int) -> Scalar[DType.uint32]:
        return self.key[index]


@always_inline
fn key(key: UInt64) -> PRNGKey:
    return PRNGKey(key)


fn split(owned key: PRNGKey, num: Int = 2) -> List[PRNGKey]:
    """Split a PRNG key into multiple keys.

    Args:
        key: The key to split.
        num: The number of keys to split into.

    Returns:
        A list of PRNG keys.

    Examples:

    ```mojo
    var key = random123.key(123)
    var keys = random123.split(key^, 10)
    ```
    .

    """
    var bits = bits[DType.uint64](key, num)
    var keys = List(
        unsafe_pointer=bits.steal_data().bitcast[PRNGKey](),
        size=num,
        capacity=num,
    )
    return keys


fn threefry2x32[
    width: Int, rounds: Int = 20
](
    key1: UInt32,
    key2: UInt32,
    ks: UInt32,
    inout count1: SIMD[DType.uint32, width],
    inout count2: SIMD[DType.uint32, width],
):
    alias rotations = InlineList[Int, 8](13, 15, 26, 6, 17, 29, 16, 24)

    @parameter
    for i in range(1, rounds + 1):
        count1 += count2
        count2 = rotate_bits_left[shift = rotations[i % 8 - 1]](count2)
        count2 ^= count1

        @parameter
        if i % 4 == 0:

            @parameter
            if (i // 4) % 3 == 1:
                count1 += key2
                count2 += ks

            @parameter
            if (i // 4) % 3 == 2:
                count1 += ks
                count2 += key1

            @parameter
            if (i // 4) % 3 == 0:
                count1 += key1
                count2 += key2

            count2 += i // 4


fn threefry2x32(
    key1: UInt32,
    key2: UInt32,
) -> SIMD[DType.uint32, 2]:
    var count = SIMD[DType.uint32, 2](0)
    threefry2x32(key1, key2, key1 ^ key2 ^ 0x1BD11BDA, count[0], count[1])
    return count


alias simd_n = 4
alias simd_width = simdwidthof[DType.uint32]() * simd_n


fn get_counts(size: Int) -> List[SIMD[DType.uint32, simd_width]]:
    var counts = List[SIMD[DType.uint32, simd_width]](capacity=size)
    counts.resize(size)

    @parameter
    fn closure(i: Int):
        counts[i] = iota[DType.uint32, simd_width](i * simd_width)

    parallelize[closure](size)

    return counts


fn _random_bits(
    key: PRNGKey, nbits: Int
) -> List[SIMD[DType.uint32, simd_width]]:
    # generate random bits, rounded up to the nearest multiple of 2 * 32 * simd_width
    var max_count = ceildiv(nbits, 2 * 32 * simd_width)
    max_count *= 2  # max_count must be even
    var counts = get_counts(max_count)
    var ks = key[0] ^ key[1] ^ 0x1BD11BDA

    @parameter
    fn closure(i: Int):
        threefry2x32(key[0], key[1], ks, counts[2 * i], counts[2 * i + 1])

    parallelize[closure](max_count // 2)
    return counts


fn bits[dtype: DType](owned key: PRNGKey, size: Int) -> List[Scalar[dtype]]:
    """Generate uniformly distributed random bits as unsigned integers.

    Parameters:
        dtype: The dtype of the output.

    Args:
        key: The key to use for the PRNG.
        size: The number of random bits to generate.

    Returns:
        A list of random unsigned integers.


    Constraints:
        - `dtype` must be an unsigned integer dtype.

    Examples:

    ```mojo
    var key = random123.key(123)
    var x = random123.bits[DType.uint32](key^, 10)
    ```
    .

    """
    constrained[
        dtype.is_unsigned(),
        "dtype must be an unsigned integer dtype",
    ]()

    var random_bits = _random_bits(key, bitwidthof[dtype]() * size)
    var res = recast[dtype, 1](random_bits^, size)
    return res


fn _uniform[
    dtype: DType,
    new_simd_width: Int = simdwidthof[dtype]() * simd_n,
](
    owned key: PRNGKey,
    size: Int,
    low: Scalar[dtype] = 0,
    high: Scalar[dtype] = 1,
) -> List[SIMD[dtype, new_simd_width]]:
    alias uint_dtype = _uint_dtype[bitwidthof[dtype]()]()
    alias mantissa_mask = FPUtils[dtype].mantissa_mask()
    alias one = SIMD[dtype, new_simd_width](1.0)
    alias exponent_one = bitcast[uint_dtype, new_simd_width](one)

    var nbits = bitwidthof[uint_dtype]() * size
    var random_bits = _random_bits(key, nbits)
    var uint_bits = recast[uint_dtype, new_simd_width](random_bits^)
    var span = high - low

    # map integer bits to floating-point numbers in [1,2)
    @parameter
    fn closure1(i: Int):
        uint_bits[i] = exponent_one | (uint_bits[i] & mantissa_mask)

    parallelize[closure1](len(uint_bits))

    var float_bits = recast[dtype, new_simd_width](uint_bits^)

    @parameter
    fn closure2(i: Int):
        float_bits[i] -= 1.0  # [1,2) -> [0,1)
        float_bits[i] = low + span * float_bits[i]

    parallelize[closure2](len(float_bits))

    return float_bits


fn uniform[
    dtype: DType
](
    owned key: PRNGKey,
    size: Int,
    low: Scalar[dtype] = 0,
    high: Scalar[dtype] = 1,
) -> List[Scalar[dtype]]:
    """Generate uniformly distributed random numbers in the range [low, high).

    Parameters:
        dtype: The dtype of the output.

    Args:
        key: The key to use for the PRNG.
        size: The number of random numbers to generate.
        low: The lower bound of the range.
        high: The upper bound of the range.

    Returns:
        A list of random numbers.

    Constraints:
        - `dtype` must be a floating-point dtype.

    Examples:

    ```mojo
    var key = random123.key(123)
    var x = random123.uniform[DType.float32](key^, 10, -1, 1)
    ```
    .
    """

    constrained[
        dtype.is_floating_point(),
        "dtype must be a floating-point dtype",
    ]()

    var float_bits = _uniform[dtype](key, size, low, high)
    var res = recast[dtype, 1](float_bits^, size)
    return res


fn normal[
    dtype: DType
](
    owned key: PRNGKey,
    size: Int,
    mean: Scalar[dtype] = 0,
    std: Scalar[dtype] = 1,
) -> List[Scalar[dtype]]:
    """Generate normally distributed random numbers.

    Parameters:
        dtype: The dtype of the output.

    Args:
        key: The key to use for the PRNG.
        size: The number of random numbers to generate.
        mean: The mean of the distribution.
        std: The standard deviation of the distribution.

    Returns:
        A list of random numbers.

    Constraints:
        - `dtype` must be a floating-point dtype.

    Examples:

    ```mojo
    var key = random123.key(0)
    var x = random123.normal[DType.float32](key^, 10, 0, 1)
    ```
    .
    """

    constrained[
        dtype.is_floating_point(),
        "dtype must be a floating-point dtype",
    ]()

    @parameter
    fn _erfinv[
        width: Int = simdwidthof[dtype]() * simd_n
    ](a: SIMD[DType.float32, width]) -> SIMD[DType.float32, width]:
        #  https://stackoverflow.com/questions/27229371/inverse-error-function-in-c#answer-49743348
        var t = fma(a, Float32(0.0) - a, Float32(1.0))
        t = log(t)

        var mask = abs(t) > 6.125

        var p_true = polynomial_evaluate[
            List[SIMD[DType.float32, width]](
                8.40016484e-1,
                -2.64646143e-1,
                4.83185798e-3,
                3.02698812e-3,
                3.93552968e-4,
                2.84108955e-5,
                1.22150334e-6,
                2.93243101e-8,
                3.03697567e-10,
            )
        ](t)

        var p_false = polynomial_evaluate[
            List[SIMD[DType.float32, width]](
                8.86226892e-1,
                -2.32015476e-1,
                1.15392581e-2,
                2.31468678e-3,
                -1.47697632e-4,
                -5.61530760e-5,
                1.12963626e-7,
                1.22774793e-6,
                1.43285448e-7,
                5.43877832e-9,
            )
        ](t)

        var p = mask.select(p_true, p_false)
        return a * p

    var low: Scalar[dtype]

    @parameter
    if dtype.is_half_float():
        low = above_minus_one[dtype]()
    else:
        low = nextafter(Scalar[dtype](-1), Scalar[dtype](0))
    alias high = Scalar[dtype](1)

    var sqrt_2 = sqrt(SIMD[dtype, simdwidthof[dtype]() * 4](2))
    var samples = _uniform[dtype](key, size, low=low, high=high)

    @parameter
    fn closure(i: Int):
        samples[i] = (
            sqrt_2 * _erfinv(samples[i].cast[DType.float32]()).cast[dtype]()
        )
        samples[i] = fma(samples[i], std, mean)

    parallelize[closure](len(samples))
    return recast[dtype, 1](samples^, size)
