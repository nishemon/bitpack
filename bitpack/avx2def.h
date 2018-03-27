#pragma once

#include <immintrin.h>

#define VEC(f)		_mm256_ ## f
#define WIDTH(f)	_mm256_ ## f ## _si256
#define UNPACK(w)	avx2_unpack ## w
#define PACK(w)		avx2_pack ## w
#define AND(a, b)		WIDTH(and)(a, b)
#define ANDNOT(a, b)	WIDTH(andnot)(a, b)
#define OR(a, b)		WIDTH(or)(a, b)
#define XOR(a, b)		WIDTH(xor)(a, b)
#define SRLI(v, s)		((s) ? VEC(srli_epi64)(v, s) : v)
#define SLLI(v, s)		((s) ? VEC(slli_epi64)(v, s) : v)
#define SRLI16(v, s)	((s) ? VEC(srli_epi16)(v, s) : v)
#define SLLI16(v, s)	((s) ? VEC(slli_epi16)(v, s) : v)
#define SRLI32(v, s)	((s) ? VEC(srli_epi32)(v, s) : v)
#define SLLI32(v, s)	((s) ? VEC(slli_epi32)(v, s) : v)
#define REGBITS		256
#define GUARD(x)
#define SLICE2(x, shift)	_mm256_or_si256(x) << (shift), x)
#define SLICE4(x, shift)	_mm256_set_epi64x((x) << ((shift) * 3), (x) << ((shift) * 2, (x) << (shift), x)

typedef __m256i VECTOR;
typedef __m128i HALF_VECTOR;
