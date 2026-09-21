M55 long-audio memory-pass sprint

Requires prepared M53 384-frame fixture/bundle.
Applies source patch, rebuilds, bit-exact tests parallel leaky-copy, then runs:
1) parallel Add / leaky-copy 4-way balanced A/B
2) 2-D time tile 504/1008/2016/4032 balanced sweep
3) RANGE_T24 long-audio retest
4) winner profile
5) winner 4.458s E2E x3
