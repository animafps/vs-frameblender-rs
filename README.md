# vs-frameblender-rs

Rust version of [couleurm/vs-frameblender](https://github.com/couleurm/vs-frameblender) using [rustsynth](https://github.com/animafps/rustsynth) with avx2 optimisation and fixed point magic

```txt
Benchmark 1: vspipe rs.vpy /dev/null
  Time (mean ± σ):      3.630 s ±  0.048 s    [User: 13.829 s, System: 2.282 s]
  Range (min … max):    3.560 s …  3.714 s    10 runs
 
Benchmark 2: vspipe cpp.vpy /dev/null
  Time (mean ± σ):      8.825 s ±  0.097 s    [User: 58.111 s, System: 1.940 s]
  Range (min … max):    8.700 s …  8.957 s    10 runs
 
Summary
  vspipe rs.vpy /dev/null ran
    2.43 ± 0.04 times faster than vspipe cpp.vpy /dev/null
```

## Usage

`frameblenderrs.Blend(clip: vnode, weights: float[]) -> clip: vnode`
