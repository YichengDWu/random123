from math.polynomial import polynomial_evaluate
from math import log, sqrt, fma


fn recast[
    new_type: DType,
    new_width: Int,
    old_type: DType,
    old_width: Int,
](
    owned l: List[SIMD[old_type, old_width]], new_size: Optional[Int] = None
) -> List[SIMD[new_type, new_width]]:
    var size: Int
    if new_size:
        size = new_size.value()
    else:
        alias multipler = (sizeof[SIMD[old_type, old_width]]() * 2) // sizeof[
            SIMD[new_type, new_width]
        ]()
        size = (len(l) // 2) * multipler
    var new_list = List(
        unsafe_pointer=l.steal_data().bitcast[SIMD[new_type, new_width]](),
        size=size,
        capacity=size,
    )
    return new_list


fn _uint_dtype[nbits: Int]() -> DType:
    @parameter
    if nbits == 8:
        return DType.uint8
    elif nbits == 16:
        return DType.uint16
    elif nbits == 32:
        return DType.uint32
    else:
        constrained[
            nbits == 64,
            "nbits must be 8, 16, 32, or 64",
        ]()
        return DType.uint64


@always_inline("nodebug")
fn below_one[dt: DType]() -> Scalar[dt]:
    constrained[dt.is_half_float()]()
    alias f = bitcast[DType.uint16, 1](Scalar[dt](1.0))
    alias m = f - 1
    return bitcast[dt, 1](m)


@always_inline("nodebug")
fn above_minus_one[dt: DType]() -> Scalar[dt]:
    constrained[dt.is_half_float()]()
    alias f = bitcast[DType.uint16, 1](Scalar[dt](-1.0))
    alias m = f - 1
    return bitcast[dt, 1](m)
