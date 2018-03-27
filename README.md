# bitpack

challenge to write fast decoding bitpack

## idea
Use `((A << n) ^ B) & MASK(x)` instead of `((A & MASK(x - n)) << n) | (B & MASK(n))` (`MASK(n) == (1 << n) - 1`).

