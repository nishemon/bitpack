#pragma once

#define UNPACK(w)		int_unpack ## w
#define PACK(w)			int_pack ## w
#define AND(a, b)		((a) & (b))
#define ANDNOT(a, b)	(~(a) & (b))
#define OR(a, b)		((a) | (b))
#define XOR(a, b)		((a) ^ (b))
#define SRLI(v, s)		((v) >> (s)) 
#define SLLI(v, s)		((v) << (s))
#define SRLI16(v, s)	(((v) >> (s)) & mem_mask[])
#define SLLI16(v, s)	(((v) << (s)) & mem_mask[16])
#define SRLI32(v, s)	(((v) >> (s))
#define SLLI32(v, s)	(((v) << (s)) & mem_mask[32])
#define REGBITS		64
#define GUARD(x)
#define VSCALE		(STRIDE/REGBITS)
#define SLICE2(x, shift)
#define SLICE4(x, shift)


typedef unsigned long long VECTOR;
typedef unsigned long HALF_VECTOR;
typedef unsigned short QUARTER_VECTOR;

static inline VECTOR srli16(VECTOR v, int s) {
	if (s < 9) {
		return (v >> s) & *(VECTOR*)&mem_mask[16 - s];
	}
	return (v >> s) & *(VECTOR*)&mem_mask[8] & *(VECTOR*)&mem_mask[16 - s];
}

static inline VECTOR slli16(VECTOR v, int s) {
	if (s < 9) {
		return (v >> s) & *(VECTOR*)&mem_mask[16 - s];
	}
	return (v >> s) & *(VECTOR*)&mem_mask[8] & *(VECTOR*)&mem_mask[16 - s];
}
