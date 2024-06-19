from benchmark import Bench, Bencher, BenchId, keep, BenchConfig, Unit, run
from random123 import *


alias size = 2**20

@parameter
fn bench_random1[
    dtype: DType, //,
    func: fn(
        owned PRNGKey, Int, Scalar[dtype], Scalar[dtype]
    ) -> List[Scalar[dtype]]
](inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        var k = random123.key(123)
        _ = func(k^, size, 0.0, 1.0)

    b.iter[call_fn]()

@parameter
fn bench_random2[
    dtype: DType, //,
    func: fn(
        owned PRNGKey, Int
    ) -> List[Scalar[dtype]]
](inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        var k = random123.key(123)
        _ = func(k^, size)

    b.iter[call_fn]()


def main():
    var m = Bench(BenchConfig(num_repetitions=1, warmup_iters=100))
    m.bench_function[bench_random2[bits[DType.uint32]]](
        BenchId("bench_bits_uint32")
    )
    m.bench_function[bench_random2[bits[DType.uint64]]](
        BenchId("bench_bits_uint64")
    )
    m.bench_function[bench_random1[uniform[DType.float32]]](
        BenchId("bench_uniform_float32")
    )
    m.bench_function[bench_random1[uniform[DType.float64]]](
        BenchId("bench_uniform_float64")
    )
    m.bench_function[bench_random1[normal[DType.float32]]](
        BenchId("bench_normal_float32")
    )
    m.bench_function[bench_random1[normal[DType.float64]]](
        BenchId("bench_normal_float64")
    )
    m.dump_report()