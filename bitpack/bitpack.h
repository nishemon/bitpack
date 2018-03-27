#include <immintrin.h>

#ifndef INLINE
#define INLINE		inline
#endif

#ifndef OR3
#define OR3(a, b, c)		OR(a, OR(b, c))
#endif
#ifndef OR4
#define OR4(a, b, c, d)		OR(OR(a, b), OR(c, d))
#endif
#ifndef XOR3
#define XOR3(a, b, c)		XOR(a, XOR(b, c))
#endif
#ifndef XOR4
#define XOR4(a, b, c, d)	XOR(XOR(a, b), XOR(c, d))
#endif


#define LD_MASK(b)	VECTOR mask ## b = *(VECTOR*)&mem_mask[b]
#ifndef LD_ZERO
#define LD_ZERO	VECTOR zero = *(VECTOR*)&mem_mask[0];
#endif

#define LD_ZERO	VECTOR zero = _mm_setzero_si128;
#define START_UNPACK(n) 	\
	const VECTOR* pk = (const VECTOR*)bits; \
	VECTOR* rw = (VECTOR*)ints; \
	LD_MASK(n);
#define START_HALF_UNPACK(n) 	\
	const HALF_VECTOR* pk = (const HALF_VECTOR*)bits; \
	VECTOR* rw = (VECTOR*)ints; \
	LD_MASK(n);
#define START_QUARTER_UNPACK(n) 	\
	const QUARTER_VECTOR* pk = (const QUARTER_VECTOR*)bits; \
	VECTOR* rw = (VECTOR*)ints; \
	LD_MASK(n);
#define START_PACK(n) 	\
	VECTOR* pk = (VECTOR*)bits; \
	const VECTOR* rw = (const VECTOR*)ints; \
	GUARD(n)
#define B(o)		pk[o * VSCALE + p]
#define I(o)		rw[o * VSCALE + p]
#define VSCALE		(STRIDE/REGBITS)


// _m256i, _m128i
static const unsigned char mem_mask[32][32] = {
	{
		0
	},{
		0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,
		0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,
		0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,
		0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,
	},{
		0x03,0x03,0x03,0x03,0x03,0x03,0x03,0x03,
		0x03,0x03,0x03,0x03,0x03,0x03,0x03,0x03,
		0x03,0x03,0x03,0x03,0x03,0x03,0x03,0x03,
		0x03,0x03,0x03,0x03,0x03,0x03,0x03,0x03,
	},{
		0x07,0x07,0x07,0x07,0x07,0x07,0x07,0x07,
		0x07,0x07,0x07,0x07,0x07,0x07,0x07,0x07,
		0x07,0x07,0x07,0x07,0x07,0x07,0x07,0x07,
		0x07,0x07,0x07,0x07,0x07,0x07,0x07,0x07,
	},{
		0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,
		0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,
		0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,
		0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,0x0F,
	},{
		0x1F,0x1F,0x1F,0x1F,0x1F,0x1F,0x1F,0x1F,
		0x1F,0x1F,0x1F,0x1F,0x1F,0x1F,0x1F,0x1F,
		0x1F,0x1F,0x1F,0x1F,0x1F,0x1F,0x1F,0x1F,
		0x1F,0x1F,0x1F,0x1F,0x1F,0x1F,0x1F,0x1F,
	},{
		0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,
		0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,
		0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,
		0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,0x3F,
	},{
		0x7F,0x7F,0x7F,0x7F,0x7F,0x7F,0x7F,0x7F,
		0x7F,0x7F,0x7F,0x7F,0x7F,0x7F,0x7F,0x7F,
		0x7F,0x7F,0x7F,0x7F,0x7F,0x7F,0x7F,0x7F,
		0x7F,0x7F,0x7F,0x7F,0x7F,0x7F,0x7F,0x7F,
	},{
		0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,
		0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,
		0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,
		0xFF,0x00,0xFF,0x00,0xFF,0x00,0xFF,0x00,
	},{
		0xFF,0x01,0xFF,0x01,0xFF,0x01,0xFF,0x01,
		0xFF,0x01,0xFF,0x01,0xFF,0x01,0xFF,0x01,
		0xFF,0x01,0xFF,0x01,0xFF,0x01,0xFF,0x01,
		0xFF,0x01,0xFF,0x01,0xFF,0x01,0xFF,0x01,
	},{
		0xFF,0x03,0xFF,0x03,0xFF,0x03,0xFF,0x03,
		0xFF,0x03,0xFF,0x03,0xFF,0x03,0xFF,0x03,
		0xFF,0x03,0xFF,0x03,0xFF,0x03,0xFF,0x03,
		0xFF,0x03,0xFF,0x03,0xFF,0x03,0xFF,0x03,
	},{
		0xFF,0x07,0xFF,0x07,0xFF,0x07,0xFF,0x07,
		0xFF,0x07,0xFF,0x07,0xFF,0x07,0xFF,0x07,
		0xFF,0x07,0xFF,0x07,0xFF,0x07,0xFF,0x07,
		0xFF,0x07,0xFF,0x07,0xFF,0x07,0xFF,0x07,
	},{
		0xFF,0x0F,0xFF,0x0F,0xFF,0x0F,0xFF,0x0F,
		0xFF,0x0F,0xFF,0x0F,0xFF,0x0F,0xFF,0x0F,
		0xFF,0x0F,0xFF,0x0F,0xFF,0x0F,0xFF,0x0F,
		0xFF,0x0F,0xFF,0x0F,0xFF,0x0F,0xFF,0x0F,
	},{
		0xFF,0x1F,0xFF,0x1F,0xFF,0x1F,0xFF,0x1F,
		0xFF,0x1F,0xFF,0x1F,0xFF,0x1F,0xFF,0x1F,
		0xFF,0x1F,0xFF,0x1F,0xFF,0x1F,0xFF,0x1F,
		0xFF,0x1F,0xFF,0x1F,0xFF,0x1F,0xFF,0x1F,
	},{
		0xFF,0x3F,0xFF,0x3F,0xFF,0x3F,0xFF,0x3F,
		0xFF,0x3F,0xFF,0x3F,0xFF,0x3F,0xFF,0x3F,
		0xFF,0x3F,0xFF,0x3F,0xFF,0x3F,0xFF,0x3F,
		0xFF,0x3F,0xFF,0x3F,0xFF,0x3F,0xFF,0x3F,
	},{
		0xFF,0x7F,0xFF,0x7F,0xFF,0x7F,0xFF,0x7F,
		0xFF,0x7F,0xFF,0x7F,0xFF,0x7F,0xFF,0x7F,
		0xFF,0x7F,0xFF,0x7F,0xFF,0x7F,0xFF,0x7F,
		0xFF,0x7F,0xFF,0x7F,0xFF,0x7F,0xFF,0x7F,
	},{
		0xFF,0xFF,0x00,0x00,0xFF,0xFF,0x00,0x00,
		0xFF,0xFF,0x00,0x00,0xFF,0xFF,0x00,0x00,
		0xFF,0xFF,0x00,0x00,0xFF,0xFF,0x00,0x00,
		0xFF,0xFF,0x00,0x00,0xFF,0xFF,0x00,0x00,
	},{
		0xFF,0xFF,0x01,0x00,0xFF,0xFF,0x01,0x00,
		0xFF,0xFF,0x01,0x00,0xFF,0xFF,0x01,0x00,
		0xFF,0xFF,0x01,0x00,0xFF,0xFF,0x01,0x00,
		0xFF,0xFF,0x01,0x00,0xFF,0xFF,0x01,0x00,
	},{
		0xFF,0xFF,0x03,0x00,0xFF,0xFF,0x03,0x00,
		0xFF,0xFF,0x03,0x00,0xFF,0xFF,0x03,0x00,
		0xFF,0xFF,0x03,0x00,0xFF,0xFF,0x03,0x00,
		0xFF,0xFF,0x03,0x00,0xFF,0xFF,0x03,0x00,
	},{
		0xFF,0xFF,0x07,0x00,0xFF,0xFF,0x07,0x00,
		0xFF,0xFF,0x07,0x00,0xFF,0xFF,0x07,0x00,
		0xFF,0xFF,0x07,0x00,0xFF,0xFF,0x07,0x00,
		0xFF,0xFF,0x07,0x00,0xFF,0xFF,0x07,0x00,
	},{
		0xFF,0xFF,0x0F,0x00,0xFF,0xFF,0x0F,0x00,
		0xFF,0xFF,0x0F,0x00,0xFF,0xFF,0x0F,0x00,
		0xFF,0xFF,0x0F,0x00,0xFF,0xFF,0x0F,0x00,
		0xFF,0xFF,0x0F,0x00,0xFF,0xFF,0x0F,0x00,
	},{
		0xFF,0xFF,0x1F,0x00,0xFF,0xFF,0x1F,0x00,
		0xFF,0xFF,0x1F,0x00,0xFF,0xFF,0x1F,0x00,
		0xFF,0xFF,0x1F,0x00,0xFF,0xFF,0x1F,0x00,
		0xFF,0xFF,0x1F,0x00,0xFF,0xFF,0x1F,0x00,
	},{
		0xFF,0xFF,0x3F,0x00,0xFF,0xFF,0x3F,0x00,
		0xFF,0xFF,0x3F,0x00,0xFF,0xFF,0x3F,0x00,
		0xFF,0xFF,0x3F,0x00,0xFF,0xFF,0x3F,0x00,
		0xFF,0xFF,0x3F,0x00,0xFF,0xFF,0x3F,0x00,
	},{
		0xFF,0xFF,0x7F,0x00,0xFF,0xFF,0x7F,0x00,
		0xFF,0xFF,0x7F,0x00,0xFF,0xFF,0x7F,0x00,
		0xFF,0xFF,0x7F,0x00,0xFF,0xFF,0x7F,0x00,
		0xFF,0xFF,0x7F,0x00,0xFF,0xFF,0x7F,0x00,
	},{
		0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,
		0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,
		0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,
		0xFF,0xFF,0xFF,0x00,0xFF,0xFF,0xFF,0x00,
	}
};


typedef void (*Unpack)(const void* bits, void* ints);
typedef void (*Pack)(const void*ints, void* bits);

// 33
INLINE void UNPACK(1)(const void* bits, void* ints) {
	START_UNPACK(1);
	for (int p = 0; p < min(STRIDE, BLOCKCOUNT) / REGBITS; p++) {
		VECTOR v;
		v = B(0);
		I(0) = AND(SRLI(v, 0), mask1);
		I(1) = AND(SRLI(v, 1), mask1);
		I(2) = AND(SRLI(v, 2), mask1);
		I(3) = AND(SRLI(v, 3), mask1);
		I(4) = AND(SRLI(v, 4), mask1);
		I(5) = AND(SRLI(v, 5), mask1);
		I(6) = AND(SRLI(v, 6), mask1);
		I(7) = AND(SRLI(v, 7), mask1);
	}
	GUARD(1);
}

INLINE void PACK(1)(const void* ints, void* bits) {
	START_PACK(1);
	for (int p = 0; p < min(STRIDE, BLOCKCOUNT) / REGBITS; p++) {
		VECTOR v1, v2, v3, v4;
		v1 = OR(SLLI(I(0), 0), SLLI(I(1), 1));
		v2 = OR(SLLI(I(2), 2), SLLI(I(3), 3));
		v3 = OR(SLLI(I(4), 4), SLLI(I(5), 5));
		v4 = OR(SLLI(I(6), 6), SLLI(I(7), 7));
		B(0) = OR4(v1, v2, v3, v4);
	}
}

#if REGBITS <= BLOCKCOUNT *2
// 32
INLINE void UNPACK(2)(const void* bits, void* ints) {
	START_UNPACK(2);
	for (int p = 0; p < VSCALE; p++) {
		VECTOR v;
		v = B(0);
		I(0) = AND(SRLI(v, 0), mask2);
		I(1) = AND(SRLI(v, 2), mask2);
		I(2) = AND(SRLI(v, 4), mask2);
		I(3) = AND(SRLI(v, 6), mask2);
	}
	GUARD(2);
}

INLINE void PACK(2)(const void* ints, void* bits) {
	START_PACK(2);
	for (int p = 0; p < VSCALE; p++) {
		VECTOR v1, v2;
		v1 = OR(SLLI(I(0), 0), SLLI(I(1), 2));
		v2 = OR(SLLI(I(2), 4), SLLI(I(3), 6));
		B(0) = OR(v1, v2);
	}
}
#else
#endif

#if 2 <= STRIDE/REGBITS
// LCM(3,4)/3 + LCM(3,4)/8 - 1 = 10
// 37
INLINE void UNPACK(3)(const void* bits, void* ints) {
	START_UNPACK(3);

	for (int p = 0; p < VSCALE; p++) {
		// 11.6
		VECTOR iv1, iv2, iv3;
		iv1 = B(0);
		iv2 = B(1);
		iv3 = B(2);
		// {3+3+2}*2,{3+3+1+1}
		I(0) = AND(iv1, mask3);
		I(1) = AND(iv2, mask3);
		I(2) = AND(SRLI(iv1, 3), mask3);
		I(3) = AND(SRLI(iv2, 3), mask3);
		I(4) = AND(iv3, mask3);
		I(5) = AND(SRLI(iv3, 4), mask3);
		I(6) = AND(XOR(SRLI(iv1, 6), SRLI(iv3, 1)), mask3);
		I(7) = AND(XOR(SRLI(iv2, 6), SRLI(iv3, 5)), mask3);
	}
	GUARD(3);
}
#elif STRIDE == REGBITS
INLINE void UNPACK(3)(const void* bits, void* ints) {
	START_HALF_UNPACK(3);

	VECTOR v1, v2;
	v1 = *(VECTOR*)&B(0);
	v2 = TWICE(B(1), 4);
	I(0) = AND(v1, mask3);
	I(1) = AND(SRLI(v1, 3), mask3);
	I(2) = AND(v2, mask3);
	I(3) = AND(XOR(SRLI(v1, 6), SRLI(v2, 1)), mask3);
	GUARD(3, op);
}
#else
#error
#endif

INLINE void PACK(3)(const void* ints, void* bits) {
	LD_MASK(2);
	LD_MASK(3);
	START_PACK(3);

	VECTOR xmask = XOR(mask3, mask2);
	for (int p = 0; p < VSCALE; p++) {
		VECTOR v1, v2, v3, v4, v5, v6;
		v1 = OR(I(0), SLLI(I(2), 3));
		v2 = OR(I(1), SLLI(I(3), 3));
		v3 = OR(I(4), SLLI(I(5), 4));
		B(0) = OR(v1, SLLI(AND(XOR(I(6), SRLI(I(4), 1)), mask2), 6));
		B(1) = OR(v2, SLLI(AND(XOR(I(7), SRLI(I(5), 1)), mask2), 6));
		v4 = SLLI(AND(XOR(SRLI(I(0), 6), I(6)), xmask), 1);
		v5 = SLLI(AND(XOR(SRLI(I(0), 6), I(7)), xmask), 5);
		B(2) = OR3(v5, v4, v3);
	}
}

// 30
INLINE void UNPACK(4)(const void* bits, void* ints) {
	START_UNPACK(4);
	// 14
	for (int p = 0; p < VSCALE; p++) {
		VECTOR v1 = B(0);
		I(0) = AND(v1, mask4);
		I(1) = AND(SRLI(v1, 4), mask4);
	}
	GUARD(4);
}
INLINE void PACK(4)(const void* ints, void* bits) {
	START_PACK(4);
	for (int p = 0; p < VSCALE; p++) {
		B(0) = OR(SLLI(I(1), 4), I(0));
	}
}

#if STRIDE/REGBITS == 2
// 40
INLINE void UNPACK(5)(const void* bits, void* ints) {
	LD_MASK(2);
	START_UNPACK(5);
	for (int p = 0; p < VSCALE; p++) {
		VECTOR iv1, iv2, iv3, iv4, iv5;
		iv1 = B(0);
		iv2 = B(1);
		iv3 = B(2);
		iv4 = B(3);
		iv5 = B(4);
#if 1
		I(0) = AND(iv1, mask5);
		I(1) = AND(iv2, mask5);
		I(2) = AND(iv3, mask5);
		I(3) = AND(iv4, mask5);
#if 1
		// {3+5}*4,{2+2+2+2}
		// 13.5
		iv1 = XOR(iv1, I(0));
		iv2 = XOR(iv2, I(1));
		iv3 = XOR(iv3, I(2));
		iv4 = XOR(iv4, I(3));
		I(4) = OR(SRLI(iv1, 3), AND(SRLI(iv5, 0), mask2));
		I(5) = OR(SRLI(iv2, 3), AND(SRLI(iv5, 2), mask2));
		I(6) = OR(SRLI(iv3, 3), AND(SRLI(iv5, 4), mask2));
		I(7) = OR(SRLI(iv4, 3), AND(SRLI(iv5, 6), mask2));
#else
		// 13.8
		op[4] = AND(XOR(SRLI(iv1, 3), iv5), mask5);
		op[5] = AND(XOR(SRLI(iv2, 3), SRLI(iv5, 2)), mask5);
		op[6] = AND(XOR(SRLI(iv3, 3), SRLI(iv5, 4)), mask5);
		op[7] = AND(XOR(SRLI(iv4, 3), SRLI(iv5, 6)), mask5);
#endif
#else
		// 18
		op[0] = AND(XOR(iv1, SRLI(iv5, 4)), mask5);
		op[1] = AND(XOR(SRLI(iv1, 4), SRLI(iv5, 3)), mask5);
		op[2] = AND(XOR(iv2, SRLI(iv5, 2)), mask5);
		op[3] = AND(XOR(SRLI(iv2, 4), SRLI(iv5, 1)), mask5);
		op[4] = AND(XOR(iv3, iv5), mask5);
		op[5] = AND(XOR(SRLI(iv3, 4), SLLI(iv5, 1)), mask5);
		op[6] = AND(XOR(iv4, SLLI(iv5, 2)), mask5);
		op[7] = AND(XOR(SRLI(iv4, 4), SLLI(iv5, 3)), mask5);
#endif
	}
	GUARD(5);
}
#elif STRIDE == REGBITS
INLINE void UNPACK(5)(const void* bits, void* ints) {
	LD_MASK(2);
	HALF_SETUP(5);

	VECTOR v1, v2, v3;
	v1 = *(VECTOR*)&HB(0);
	v2 = *(VECTOR*)&HB(2);
	v3 = TWICE(HB(3), 4);
	I(0) = AND(v1, mask5);
	I(1) = AND(v2, mask5);
	VECTOR tv1, tv2;
	tv1 = XOR(v1, I(0));
	tv2 = XOR(v2, I(1));
	I(2) = OR(SRLI(tv1, 3), AND(SRLI(iv5, 0), mask2));
	I(3) = OR(SRLI(tv2, 3), AND(SRLI(iv5, 2), mask2));

	GUARD(5, op);
}
#else
#error
#endif


INLINE void PACK(5)(const void* ints, void* bits) {
	LD_MASK(2);
	LD_MASK(5);
	START_PACK(5);
	VECTOR xmask = XOR(mask2, mask5);
	for (int p = 0; p < VSCALE; p++) {
		VECTOR iv1, iv2, iv3, iv4, iv5;
		B(0) = OR(I(0), SLLI(AND(I(4), xmask), 3));
		B(1) = OR(I(1), SLLI(AND(I(5), xmask), 3));
		B(2) = OR(I(2), SLLI(AND(I(6), xmask), 3));
		B(3) = OR(I(3), SLLI(AND(I(7), xmask), 3));
		B(4) = OR4(
			SLLI(AND(I(4), mask2), 0),
			SLLI(AND(I(5), mask2), 2),
			SLLI(AND(I(6), mask2), 4),
			SLLI(AND(I(7), mask2), 6)
		);
	}
}

//40
INLINE void UNPACK(6)(const void* bits, void* ints) {
	START_UNPACK(6);
	for (int p = 0; p < VSCALE; p++) {
		// 12.7
		VECTOR v1, v2, v3;
		v1 = B(0);
		v2 = B(1);
		v3 = B(2);
		// {2+6}*3
		I(0) = AND(v1, mask6);
		I(1) = AND(v2, mask6);
		I(2) = AND(v3, mask6);
		I(3) = AND(XOR3(SRLI(v1, 6), SRLI(v2, 4), SRLI(v3, 2)), mask6);
	}
	// TODO try {2+6}*2,{4+4}
#if 0
	// 14.6
	VECTOR iv1, iv2, iv3, iv4, iv5, iv6;
	iv1 = XOR(iv1, op[0]);
	iv2 = XOR(iv2, op[1]);
	iv3 = XOR(iv3, op[2]);
	iv4 = XOR(iv4, op[3]);
	iv5 = XOR(iv5, op[2]);
	iv6 = XOR(iv6, op[3]);
	op[6] = OR(OR(SRLI(iv1, 6), SRLI(iv3, 4)), SRLI(iv5, 2));
	op[7] = OR(OR(SRLI(iv2, 6), SRLI(iv4, 4)), SRLI(iv6, 2));
#endif
	GUARD(6);
}

INLINE void PACK(6)(const void* ints, void* bits) {
	LD_MASK(2);
	START_PACK(6);

	for (int p = 0; p < VSCALE; p++) {
		VECTOR v1, v2, v3, v4;
		v1 = I(0);
		v2 = I(1);
		v3 = I(2);
		v4 = I(3);
		B(0) = OR(v1, SLLI(AND(XOR3(v4, SRLI(v2, 4), SRLI(v3, 2)), SLLI(mask2, 0)), 6));
		B(1) = OR(v2, SLLI(AND(XOR3(SRLI(v1, 6), v4, SRLI(v3, 2)), SLLI(mask2, 2)), 4));
		B(2) = OR(v3, SLLI(AND(XOR3(SRLI(v1, 6), SRLI(v2, 4), v4), SLLI(mask2, 4)), 2));
	}
}

// 45
INLINE void UNPACK(7)(const void* bits, int* ints) {
	START_UNPACK(7);
	for (int p = 0; p < VSCALE; p++) {
		VECTOR v1, v2;
		VECTOR iv1, iv2, iv3, iv4, iv5, iv6, iv7;
		iv1 = B(0); iv2 = B(1); iv3 = B(2); iv4 = B(3);
		iv5 = B(4); iv6 = B(5); iv7 = B(6);
		// {1+7}*6, {4+4}
		// 9.1
		I(0) = AND(iv1, mask7);
		I(1) = AND(iv2, mask7);
		I(2) = AND(iv3, mask7);
		I(3) = AND(iv4, mask7);
		v1 = XOR4(SRLI(iv1, 3), SRLI(iv7, 0), SRLI(iv2, 2), SRLI(iv3, 1));
		v2 = XOR4(SRLI(iv4, 3), SRLI(iv7, 4), SRLI(iv5, 2), SRLI(iv6, 1));
		I(4) = AND(iv5, mask7);
		I(5) = AND(iv6, mask7);
		I(6) = AND(v1, mask7);
		I(7) = AND(v2, mask7);
	}
	GUARD(7);
}
// ng
INLINE void PACK(7)(const void* ints, void* bits) {
	START_PACK(7);
	LD_MASK(4);
	LD_MASK(1);
	VECTOR hmask = SLLI(mask1, 7);

	for (int p = 0; p < VSCALE; p++) {
		VECTOR v1, v2, v3, v4, v5, v6, v7;
		v1 = XOR4(SRLI(I(0), 3), I(6), SRLI(I(1), 2), SRLI(I(2), 1));
		v2 = XOR4(SRLI(I(3), 3), I(7), SRLI(I(4), 2), SRLI(I(5), 1));
		v3 = XOR(v1, SLLI(v2, 4));
		B(0) = OR(I(0), AND(SLLI(v3, 3), hmask));
		B(1) = OR(I(1), AND(SLLI(v3, 2), hmask));
		B(2) = OR(I(2), AND(SLLI(v3, 1), hmask));
		B(3) = OR(I(3), AND(SLLI(v3, 7), hmask));
		B(4) = OR(I(4), AND(SLLI(v3, 6), hmask));
		B(5) = OR(I(5), AND(SLLI(v3, 5), hmask));
		B(6) = OR(AND(v1, mask4), SLLI(AND(v2, mask4), 4));
	}
}

// 49
INLINE void UNPACK(8)(const void* bits, void* ints) {
	START_UNPACK(8);

	for (int p = 0; p < VSCALE; p++) {
		VECTOR v = B(0);
		I(0) = AND(v, mask8);
		I(1) = SRLI16(v, 8);
	}
	GUARD(8);
}

INLINE void PACK(8)(const void* ints, void* bits) {
	START_PACK(8);

	for (int p = 0; p < VSCALE; p++) {
		B(0) = OR(I(0), SLLI16(I(1), 8));
	}
}

/*
In upper 8, upper bits and lower bits of 16 bits are unpacked by 1 op,
SRLI16() and AND(, mask).
*/
// 76
INLINE void UNPACK(9)(const void* bits, void* ints) {
	START_UNPACK(9);
	LD_MASK(2);
	LD_MASK(7);
	LD_MASK(4);
	VECTOR xmask = ANDNOT(mask7, mask2);
	for (int p = 0; p < VSCALE; p++) {
		VECTOR iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9;
		iv1 = B(0); iv2 = B(1); iv3 = B(2); iv4 = B(3);
		iv5 = B(4); iv6 = B(5); iv7 = B(6); iv8 = B(7);
		iv9 = B(8);
#if 0
		// 15.2
		// high vscale
		// {7+9}*8,{2*8}
		// 16.1
		I(0) = AND(iv1, mask9);
		I(1) = AND(iv2, mask9);
		I(2) = AND(iv3, mask9);
		I(3) = AND(iv4, mask9);
		I(4) = AND(iv5, mask9);
		I(5) = AND(iv6, mask9);
		I(6) = AND(iv7, mask9);
		I(7) = AND(iv8, mask9);
		I(8)  = AND(XOR(SRLI16(iv1, 7), SRLI16(iv9, 0)), mask9);
		I(9)  = AND(XOR(SRLI16(iv2, 7), SRLI16(iv9, 2)), mask9);
		I(10) = AND(XOR(SRLI16(iv3, 7), SRLI16(iv9, 4)), mask9);
		I(11) = AND(XOR(SRLI16(iv4, 7), SRLI16(iv9, 6)), mask9);
		I(12) = XOR(SRLI16(iv5, 7), SRLI16(iv9, 8));
		I(13) = XOR(SRLI16(iv6, 7), SRLI16(iv9, 10));
		I(14) = XOR(SRLI16(iv7, 7), SRLI16(iv9, 12));
		I(15) = XOR(SRLI16(iv8, 7), SRLI16(iv9, 14));
		// 8+ 7*4 - 1 = 35
#elif 1
		//13.9
		// {7+9}*4{9+5+2}*4{4+4+4+4}
		// 26
		I(0) = AND(iv1, mask9);
		I(1) = AND(iv2, mask9);
		I(2) = AND(iv3, mask9);
		I(3) = AND(iv4, mask9);
		I(4) = AND(XOR(SRLI16(iv1, 7), iv5), mask9);
		I(5) = AND(XOR(SRLI16(iv2, 7), iv6), mask9);
		I(6) = AND(XOR(SRLI16(iv3, 7), iv7), mask9);
		I(7) = AND(XOR(SRLI16(iv4, 7), iv8), mask9);//12
		I(8) = SRLI16(iv5, 7);
		I(9) = SRLI16(iv6, 7);
		I(10) = SRLI16(iv7, 7);
		I(11) = SRLI16(iv8, 7);
		I(12) = AND(XOR(SLLI16(iv5, 2), SRLI16(iv9, 0)), mask9);
		I(13) = AND(XOR(SLLI16(iv6, 2), SRLI16(iv9, 4)), mask9);//7
		I(14) = AND(XOR(SLLI16(iv7, 2), SRLI16(iv9, 8)), mask9);
		I(15) = AND(XOR(SLLI16(iv8, 2), SRLI16(iv9, 12)), mask9);//6
		// 33
#elif 1
		//13.9
		// {7+9}*4{9+5+2}*4{4+4+4+4}
		I(0) = AND(iv1, mask9);
		I(1) = AND(iv2, mask9);
		I(2) = AND(iv3, mask9);
		I(3) = AND(iv4, mask9);
		I(4) = OR(SRLI16(XOR(iv1, I(0)), 7), AND(iv5, mask2));
		I(5) = OR(SRLI16(XOR(iv2, I(1)), 7), AND(iv6, mask2));
		I(6) = OR(SRLI16(XOR(iv3, I(2)), 7), AND(iv7, mask2));
		I(7) = OR(SRLI16(XOR(iv4, I(3)), 7), AND(iv8, mask2));
		I(8) = SRLI16(iv5, 7);
		I(9) = SRLI16(iv6, 7);
		I(10) = SRLI16(iv7, 7);
		I(11) = SRLI16(iv8, 7);
		I(12) = OR(SLLI16(AND(iv5, xmask), 2), AND(SRLI16(iv9, 0), mask4));
		I(13) = OR(SLLI16(AND(iv6, xmask), 2), AND(SRLI16(iv9, 0), mask4));
		I(14) = OR(SLLI16(AND(iv7, xmask), 2), AND(SRLI16(iv9, 0), mask4));
		I(15) = OR(SLLI16(AND(iv8, xmask), 2), AND(SRLI16(iv9, 0), mask4));
		// 33
#elif 0
		// 14.2
		// ({9+7}{9+5+2}{9+3+4}{9+1+6})*2{8+8}
		I(0) = AND(iv1, mask9);
		I(1) = AND(iv2, mask9);
		I(2) = AND(XOR(SRLI16(iv1, 2), iv3), mask9);
		I(3) = AND(XOR(SRLI16(iv2, 2), iv4), mask9);
		I(4) = SRLI16(iv3, 7);
		I(5) = SRLI16(iv4, 7);
		I(6) = AND(XOR(SLLI16(iv3, 2), iv5), mask9);
		I(7) = AND(XOR(SLLI16(iv4, 2), iv6), mask9);
		I(8) = SRLI16(iv5, 7);
		I(9) = SRLI16(iv6, 7);
		I(10) = AND(XOR(SLLI16(iv5, 2), iv7), mask9);
		I(11) = AND(XOR(SLLI16(iv6, 2), iv8), mask9);
		I(12) = SRLI16(iv7, 7);
		I(13) = SRLI16(iv8, 7); // 8
		I(14) = AND(XOR(SLLI16(iv7, 2), iv5), mask9);
		I(15) = AND(XOR(SLLI16(iv8, 2), SRLI16(iv5, 8)), mask9); //25
		// 33
#else
		// {1*16},{8+8}*8
		// 17.9
		//33
		LD_MASK(1);
		VECTOR tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8;
		tv1 = AND(iv1, mask1);
		tv2 = AND(SRLI(iv1, 1), mask1);
		tv3 = AND(SRLI(iv1, 2), mask1);
		tv4 = AND(SRLI(iv1, 3), mask1);
		tv5 = AND(SRLI(iv1, 4), mask1);
		tv6 = AND(SRLI(iv1, 5), mask1);
		tv7 = AND(SRLI(iv1, 6), mask1);
		tv8 = AND(SRLI(iv1, 7), mask1);
		I(0) = VEC(unpackhi_epi8)(iv2, tv1);
		I(1) = VEC(unpacklo_epi8)(iv2, tv1);
		I(2) = VEC(unpackhi_epi8)(iv3, tv2);
		I(3) = VEC(unpacklo_epi8)(iv3, tv2);
		I(4) = VEC(unpackhi_epi8)(iv4, tv3);
		I(5) = VEC(unpacklo_epi8)(iv4, tv3);
		I(6) = VEC(unpackhi_epi8)(iv5, tv4);
		I(7) = VEC(unpacklo_epi8)(iv5, tv4);
		I(8) = VEC(unpackhi_epi8)(iv6, tv5);
		I(9) = VEC(unpacklo_epi8)(iv6, tv5);
		I(10) = VEC(unpackhi_epi8)(iv7, tv6);
		I(11) = VEC(unpacklo_epi8)(iv7, tv6);
		I(12) = VEC(unpackhi_epi8)(iv8, tv7);
		I(13) = VEC(unpacklo_epi8)(iv8, tv7);
		I(14) = VEC(unpackhi_epi8)(iv9, tv8);
		I(15) = VEC(unpacklo_epi8)(iv9, tv8);
#endif
	}
}

INLINE void PACK(9)(const void* ints, void* bits) {
	START_PACK(9);
	LD_MASK(9);
	LD_MASK(2);
	LD_MASK(4);
	VECTOR xmask = ANDNOT(mask4, mask9);
	for (int p = 0; p < VSCALE; p++) {
		VECTOR v[4];
		v[0] = XOR(AND(I(4), mask2), SRLI16(I(0), 7));
		v[1] = XOR(AND(I(5), mask2), SRLI16(I(1), 7));
		v[2] = XOR(AND(I(6), mask2), SRLI16(I(2), 7));
		v[3] = XOR(AND(I(7), mask2), SRLI16(I(3), 7));
		B(8) = XOR4(
			SLLI16(XOR(SLLI16(v[0], 2), AND(I(12), mask4)), 0),
			SLLI16(XOR(SLLI16(v[1], 2), AND(I(13), mask4)), 4),
			SLLI16(XOR(SLLI16(v[2], 2), AND(I(14), mask4)), 8),
			SLLI16(XOR(SLLI16(v[3], 2), AND(I(15), mask4)), 12)
		);
		B(4) = OR3(SLLI16(I(8), 7), SRLI16(AND(XOR(I(12), SRLI16(B(8), 0)), xmask), 2), v[0]);
		B(5) = OR3(SLLI16(I(9), 7), SRLI16(AND(XOR(I(13), SRLI16(B(8), 4)), xmask), 2), v[1]);
		B(6) = OR3(SLLI16(I(10), 7), SRLI16(AND(XOR(I(14), SRLI16(B(8), 8)), xmask), 2), v[2]);
		B(7) = OR3(SLLI16(I(11), 7), SRLI16(AND(XOR(I(15), SRLI16(B(8), 12)), xmask), 2), v[3]);
		B(0) = OR(I(0), SLLI16(ANDNOT(mask2, XOR(I(4), B(4))), 7));
		B(1) = OR(I(1), SLLI16(ANDNOT(mask2, XOR(I(5), B(5))), 7));
		B(2) = OR(I(2), SLLI16(ANDNOT(mask2, XOR(I(6), B(6))), 7));
		B(3) = OR(I(3), SLLI16(ANDNOT(mask2, XOR(I(7), B(7))), 7));

		/*
		I(0) = AND(iv1, mask9);
		I(1) = AND(iv2, mask9);
		I(2) = AND(iv3, mask9);
		I(3) = AND(iv4, mask9);
		I(4) = AND(XOR(SRLI16(iv1, 7), iv5), mask9);
		I(5) = AND(XOR(SRLI16(iv2, 7), iv6), mask9);
		I(6) = AND(XOR(SRLI16(iv3, 7), iv7), mask9);
		I(7) = AND(XOR(SRLI16(iv4, 7), iv8), mask9);//12
		I(8) = SRLI16(iv5, 7);
		I(9) = SRLI16(iv6, 7);
		I(10) = SRLI16(iv7, 7);
		I(11) = SRLI16(iv8, 7);
		I(12) = AND(XOR(SLLI16(iv5, 2), SRLI16(iv9, 0)), mask9);
		I(13) = AND(XOR(SLLI16(iv6, 2), SRLI16(iv9, 4)), mask9);//7
		I(14) = AND(XOR(SLLI16(iv7, 2), SRLI16(iv9, 8)), mask9);
		I(15) = AND(XOR(SLLI16(iv8, 2), SRLI16(iv9, 12)), mask9);//6
*/
	}

}

// LCM(10,16)/10 + LCM(10,16)/16 - 1 = 12
// OP = (LCM(10,16)/10) * 4
//73
INLINE void UNPACK(10)(const void* bits, void* ints) {
	START_UNPACK(10);
	for (int p = 0; p < VSCALE; p++) {
		// 27.5
		VECTOR iv1, iv2, iv3, iv4, iv5;
		iv1 = B(0); iv2 = B(1); iv3 = B(2); iv4 = B(3);
		iv5 = B(4);
		// {6+10}*4,{4*4}
		I(0) = AND(iv1, mask10);
		I(1) = AND(iv2, mask10);
		I(2) = AND(iv3, mask10);
		I(3) = AND(iv4, mask10);
		I(4) = AND(XOR(SRLI16(iv1, 6), SRLI16(iv5, 0)), mask10);
		I(5) = AND(XOR(SRLI16(iv2, 6), SRLI16(iv5, 4)), mask10);//7
		I(6) = XOR(SRLI16(iv3, 6), SRLI16(iv5, 8));
		I(7) = XOR(SRLI16(iv4, 6), SRLI16(iv5, 12));// 6
	}
}

INLINE void PACK(10)(const void* ints, void* bits) {
	START_PACK(10);
	LD_MASK(10);
	LD_MASK(4);
	VECTOR xmask = XOR(mask10, mask4);
	for (int p = 0; p < VSCALE; p++) {
		VECTOR v1, v2;
		v1 = OR4(
			SLLI16(AND(I(4), mask4), 0),
			SLLI16(AND(I(5), mask4), 4),
			SLLI16(AND(I(6), mask4), 8),
			SLLI16(AND(I(7), mask4), 12)
		);
		v2 = OR4(
			SLLI16(SRLI16(I(0), 6), 0),
			SLLI16(SRLI16(I(1), 6), 4),
			SLLI16(SRLI16(I(2), 6), 8),
			SLLI16(SRLI16(I(3), 6), 12)
		);
		B(0) = OR(I(0), SLLI16(XOR(I(4), SRLI16(v1, 0)), 6));
		B(1) = OR(I(1), SLLI16(XOR(I(5), SRLI16(v1, 4)), 6));
		B(2) = OR(I(2), SLLI16(XOR(I(6), SRLI16(v1, 8)), 6));
		B(3) = OR(I(3), SLLI16(ANDNOT(mask4, I(7)), 6));
		B(4) = XOR(v1, v2);
	}
}

// LCM(11,16)/11 + LCM(11,16)/16 - 1 = 26
INLINE void UNPACK(11)(const void* bits, void* ints) {
	START_UNPACK(11);
	for (int p = 0; p < VSCALE; p++) {
		VECTOR iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, ivA, ivB;
		iv1 = B(0); iv2 = B(1); iv3 = B(2); iv4 = B(3);
		iv5 = B(4); iv6 = B(5); iv7 = B(6); iv8 = B(7);
		iv9 = B(8); ivA = B(9); ivB = B(10);

#if 0
		// 19sec
		// 83
		LD_MASK(3);
		// [UNPACK(3)],[UNPACK(8)]

		VECTOR tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8;
		tv1 = AND(iv1, mask3);
		tv2 = AND(iv2, mask3);
		tv3 = AND(SRLI(iv1, 3), mask3);
		tv4 = AND(SRLI(iv2, 3), mask3);
		tv5 = AND(iv3, mask3);
		tv6 = AND(SRLI(iv3, 3), mask3); // 9
		tv7 = AND(XOR(SRLI(iv1, 6), SRLI(iv2, 4)), mask3);
		tv8 = AND(XOR(SRLI(iv3, 5), SRLI(iv2, 7)), mask3); //8

		op[0] = VEC(unpackhi_epi8)(iv4, tv1);
		op[0] = AND(iv4, mask8), 
		op[1] = VEC(unpacklo_epi8)(iv4, tv1);
		op[2] = VEC(unpackhi_epi8)(iv5, tv2);
		op[3] = VEC(unpacklo_epi8)(iv5, tv2);
		op[4] = VEC(unpackhi_epi8)(iv6, tv3);
		op[5] = VEC(unpacklo_epi8)(iv6, tv3);
		op[6] = VEC(unpackhi_epi8)(iv7, tv4);
		op[7] = VEC(unpacklo_epi8)(iv7, tv4);
		op[8] = VEC(unpackhi_epi8)(iv8, tv5);
		op[9] = VEC(unpacklo_epi8)(iv8, tv5);
		op[10] = VEC(unpackhi_epi8)(iv9, tv6);
		op[11] = VEC(unpacklo_epi8)(iv9, tv6);
		op[12] = VEC(unpackhi_epi8)(ivA, tv7);
		op[13] = VEC(unpacklo_epi8)(ivA, tv7);
		op[14] = VEC(unpackhi_epi8)(ivB, tv8);
		op[15] = VEC(unpacklo_epi8)(ivB, tv8);
#else
		// 18sec
		// 82
		// {5+11}*10,{1*4+6+6} : cost(26)
		// alter: {5+11}*10,{1*5+11} : cost(26)
		// alter: {5+11}*8,{6+6+4}*2,{6+6+2+2}: cost(26)
		// {5+11}{6+10}{1+11+4}{7+9}{2+11+3}{8} x 2 : cost(26)
		//  S A  XA  SXA  A  SXA  SXA  A  SXA = 37
		// {5+11}*4{6+7+3}*4{8+8}*2{4+4+4+4}
		LD_MASK(8);
		I(0) = AND(iv1, mask11);
		I(1) = AND(iv2, mask11);
		I(2) = AND(iv3, mask11);
		I(3) = AND(iv4, mask11);
		I(4) = AND(iv5, mask11);
		I(5) = AND(iv6, mask11);
		I(6) = AND(iv7, mask11);
		I(7) = AND(iv8, mask11);
		I(8) = AND(iv9, mask11);
		I(9) = AND(ivA, mask11); // 10
		I(10) = XOR3(SRLI16(iv1, 11), SRLI16(iv3, 6), SRLI16(iv5, 5));
		I(11) = XOR3(SRLI16(iv2, 11), SRLI16(iv4, 6), SRLI16(iv6, 5)); // 10
		I(12) = AND(XOR3(SRLI16(iv5, 11), SRLI16(iv7, 7), SRLI16(iv9, 2)), mask11);
		I(13) = AND(XOR3(SRLI16(iv6, 11), SRLI16(iv8, 7), SRLI16(ivA, 2)), mask11); // 12
		I(14) = XOR(SRLI16(iv9, 5), AND(ivB, mask8));
		I(15) = XOR(SRLI16(ivA, 5), SRLI16(ivB, 8)); //6
		// 25 22 + 3 + 8
		// 5 + 6, 5 + 6, 5 + 4
#endif
	}
}

INLINE void PACK(11)(const void* ints, void* bits) {
	START_PACK(11);
}

INLINE void UNPACK(12)(const void* bits, void* ints) {
	LD_MASK(4);
	LD_MASK(8);
	START_UNPACK(12);
	for (int p = 0; p < VSCALE; p++) {
		VECTOR iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, ivA, ivB, ivC;

#if 0
		// for skylake, ivy (unpack)
		// 14.5
		// 17sec
		//72
		I(0) = AND(iv1, mask12);
		I(1) = AND(iv2, mask12);
		I(2) = AND(iv3, mask12);
		I(3) = AND(iv4, mask12);
		I(4) = AND(iv5, mask12);
		I(5) = AND(iv6, mask12);
		I(6) = AND(iv7, mask12);
		I(7) = AND(iv8, mask12);
		VECTOR tv1, tv2, tv3, tv4;
		tv1 = VEC(packus_epi16)(SRLI16(iv1, 12), SRLI16(iv3, 12));
		tv2 = VEC(packus_epi16)(SRLI16(iv2, 12), SRLI16(iv4, 12));
		tv3 = VEC(packus_epi16)(SRLI16(iv5, 12), SRLI16(iv7, 12));
		tv4 = VEC(packus_epi16)(SRLI16(iv6, 12), SRLI16(iv8, 12));
		I(8) = VEC(unpackhi_epi8)(iv9, tv1);
		I(9) = VEC(unpacklo_epi8)(iv9, tv1);
		I(10) = VEC(unpackhi_epi8)(ivA, tv2);
		I(11) = VEC(unpacklo_epi8)(ivA, tv2);
		I(12) = VEC(unpackhi_epi8)(ivB, tv3);
		I(13) = VEC(unpacklo_epi8)(ivB, tv3);
		I(14) = VEC(unpackhi_epi8)(ivC, tv4);
		I(15) = VEC(unpacklo_epi8)(ivC, tv4); // 28
#elif 1
		for (int i = 0; i < 4; i++) {
			iv1 = B(i * 3 + 0); iv2 = B(i * 3 + 1); iv3 = B(i * 3 + 2);
			//15.3
			// {12+4}*2,{8+8}
			//  A  SXA = 9 (6)
			I(i * 4 + 0) = AND(iv1, mask12);
			I(i * 4 + 1) = AND(iv2, mask12);
			I(i * 4 + 2) = AND(XOR(SRLI16(iv1, 4), iv3), mask12);
			//		I(2) = XOR(SRLI16(iv1, 4), AND(iv3, mask8));
			I(i * 4 + 3) = XOR(SRLI16(iv2, 4), SRLI(iv3, 8)); // 8
		}
#elif 1
		iv1 = B(0); iv2 = B(1); iv3 = B(2); iv4 = B(3);
		iv5 = B(4); iv6 = B(5); iv7 = B(6); iv8 = B(7);
		iv9 = B(8); ivA = B(9); ivB = B(10); ivC = B(11);
		I(1) = AND(iv2, mask12);
		I(2) = AND(iv3, mask12);
		I(3) = AND(iv4, mask12);
		I(4) = AND(iv5, mask12);
		I(5) = AND(iv6, mask12);
		I(6) = AND(iv7, mask12);
		I(7) = AND(iv8, mask12); // 8
		I(8) = XOR(SRLI16(iv1, 4), AND(iv9, mask8));
		I(9) = XOR(SRLI16(iv2, 4), AND(ivA, mask8));
		I(10) = OR(SRLI16(iv3, 4), SRLI(iv9, 8));
		I(11) = OR(SRLI16(iv4, 4), SRLI(ivA, 8));
		I(12) = XOR(SRLI16(iv5, 4), AND(ivB, mask8));
		I(13) = XOR(SRLI16(iv6, 4), AND(ivC, mask8));
		I(14) = OR(SRLI16(iv7, 4), SRLI(ivB, 8));
		I(15) = OR(SRLI16(iv8, 4), SRLI(ivC, 8)); // 24

#elif 0
		// 17sec
		//77
		op[0] = AND(iv1, mask12);
		op[1] = AND(iv2, mask12);
		op[2] = AND(iv3, mask12);
		op[3] = AND(iv4, mask12);
		op[4] = AND(iv5, mask12);
		op[5] = AND(iv6, mask12);
		op[6] = AND(iv7, mask12);
		op[7] = AND(iv8, mask12);
		op[8] = AND(iv9, mask12);
		op[9] = AND(ivA, mask12);
		op[10] = AND(ivB, mask12);
		op[11] = AND(ivC, mask12);
		VECTOR tv1, tv2;
		tv1 = VEC(packus_epi16)(VEC(srli_epi16)(iv1, 12), VEC(srli_epi16)(iv2, 12));
		tv2 = VEC(slli_epi16)(VEC(srli_epi16)(iv3, 12), 4);
		op[12] = OR(tv1, tv2);
		tv1 = VEC(packus_epi16)(VEC(srli_epi16)(iv4, 12), VEC(srli_epi16)(iv5, 12));
		tv2 = VEC(slli_epi16)(VEC(srli_epi16)(iv6, 12), 4);
		op[13] = OR(tv1, tv2);
		tv1 = VEC(packus_epi16)(VEC(srli_epi16)(iv7, 12), VEC(srli_epi16)(iv8, 12));
		tv2 = VEC(slli_epi16)(VEC(srli_epi16)(iv9, 12), 4);
		op[14] = OR(tv1, tv2);
		tv1 = VEC(packus_epi16)(VEC(srli_epi16)(ivA, 12), VEC(srli_epi16)(ivB, 12));
		tv2 = VEC(slli_epi16)(VEC(srli_epi16)(ivC, 12), 4);
		op[15] = OR(tv1, tv2);
#else
		// 19sec
		// 73
		VECTOR tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8;
		tv1 = AND(iv1, mask4);
		tv2 = AND(iv2, mask4);
		tv3 = AND(SRLI(iv1, 4), mask4);
		tv4 = AND(SRLI(iv2, 4), mask4);
		tv5 = AND(iv3, mask4);
		tv6 = AND(iv4, mask4);
		tv7 = AND(SRLI(iv3, 4), mask4);
		tv8 = AND(SRLI(iv4, 4), mask4);

		op[0] = VEC(unpackhi_epi8)(iv5, tv1);
		op[1] = VEC(unpacklo_epi8)(iv5, tv1);
		op[2] = VEC(unpackhi_epi8)(iv6, tv2);
		op[3] = VEC(unpacklo_epi8)(iv6, tv2);
		op[4] = VEC(unpackhi_epi8)(iv7, tv3);
		op[5] = VEC(unpacklo_epi8)(iv7, tv3);
		op[6] = VEC(unpackhi_epi8)(iv8, tv4);
		op[7] = VEC(unpacklo_epi8)(iv8, tv4);
		op[8] = VEC(unpackhi_epi8)(iv9, tv5);
		op[9] = VEC(unpacklo_epi8)(iv9, tv5);
		op[10] = VEC(unpackhi_epi8)(ivA, tv6);
		op[11] = VEC(unpacklo_epi8)(ivA, tv6);
		op[12] = VEC(unpackhi_epi8)(ivB, tv7);
		op[13] = VEC(unpacklo_epi8)(ivB, tv7);
		op[14] = VEC(unpackhi_epi8)(ivC, tv8);
		op[15] = VEC(unpacklo_epi8)(ivC, tv8);
#endif
	}
}

INLINE void PACK(12)(const void* ints, void* bits) {
	START_PACK(12);
}

INLINE void UNPACK(13)(const void* bits, void* ints) {
	START_UNPACK(13);
	for (int p = 0; p < VSCALE; p++) {
		VECTOR iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, ivA, ivB, ivC, ivD;
		iv1 = B(0); iv2 = B(1); iv3 = B(2); iv4 = B(3);
		iv5 = B(4); iv6 = B(5); iv7 = B(6); iv8 = B(7);
		iv9 = B(8); ivA = B(9); ivB = B(10); ivC = B(11);
		ivD = B(12);

		// {3+13}*4,{10+6}*4,{9+7}*4,{4+4+4+4}
		LD_MASK(8);
		I(0) = AND(iv1, mask13);
		I(1) = AND(iv2, mask13);
		I(2) = AND(iv3, mask13);
		I(3) = AND(iv4, mask13);
		I(4) = XOR(SRLI16(iv1, 3), SRLI16(iv5, 6));
		I(5) = XOR(SRLI16(iv2, 3), SRLI16(iv6, 6));
		I(6) = XOR(SRLI16(iv3, 3), SRLI16(iv7, 6));
		I(7) = XOR(SRLI16(iv4, 3), SRLI16(iv8, 6));

		I(8) = AND(XOR(SRLI16(iv5, 3), iv9), mask13);
		I(9) = AND(XOR(SRLI16(iv6, 3), ivA), mask13);
		I(10) = AND(XOR(SRLI16(iv7, 3), ivB), mask13);
		I(11) = AND(XOR(SRLI16(iv8, 3), ivC), mask13);

		I(12) = AND(XOR(SRLI16(iv9, 3), SRLI16(ivD, 0)), mask13);
		I(13) = XOR(SRLI16(ivA, 3), SRLI16(ivD, 4));
		I(14) = XOR(SRLI16(ivB, 3), SRLI16(ivD, 8));
		I(15) = XOR(SRLI16(ivC, 3), SRLI16(ivD, 12));
	}
}

INLINE void PACK(13)(const void* ints, void* bits) {
	START_PACK(13);
	for (int p = 0; p < VSCALE; p++) {
	}
}

INLINE void UNPACK(14)(const void* bits, void* ints) {
	START_UNPACK(14);
	for (int p = 0; p < VSCALE; p++) {
		// {14+2}*2,{12+4}*2,{10+6}*2,{8+8}
	}
}

INLINE void PACK(14)(const void* ints, void* bits) {
	START_PACK(14);
}

INLINE void UNPACK(15)(const void* bits, void* ints) {
	START_UNPACK(15);
	for (int p = 0; p < VSCALE; p++) {
		// {15+1}*4,{14+2}*4,{13+3}*4,{8+8}*2,{4+4+4+4}
	}
}

INLINE void PACK(15)(const void* ints, void* bits) {
	START_PACK(15);
}

// 97
INLINE void UNPACK(16)(const void* bits, void* ints) {
	START_UNPACK(16);
	for (int p = 0; p < VSCALE; p++) {
		VECTOR v = B(0);
		I(0) = AND(v, mask16);
		I(1) = SRLI32(v, 16);
	}
}

INLINE void PACK(16)(const void* ints, void* bits) {
	START_PACK(16);
	LD_MASK(16);
	for (int p = 0; p < VSCALE; p++) {
		B(0) = OR(AND(I(0), mask16), SLLI32(I(1), 16));
	}
}

INLINE void UNPACK(17)(const void* bits, void* ints) {
	START_UNPACK(17);
	for (int p = 0; p < VSCALE; p++) {
		// {17+15}*4,{2+17+13}*4,{17+4+11}*4,{17+6+9}*4,{8+8+8+8}
		// [UNPACK(9)],[UNPACK(8)]
	}
}

INLINE void PACK(17)(const void* ints, void* bits) {
	START_PACK(17);
	for (int p = 0; p < VSCALE; p++) {
	}
}

INLINE void UNPACK(18)(const void* bits, void* ints) {
	START_UNPACK(18);
	for (int p = 0; p < VSCALE; p++) {
		// {18+14}*4,{18+4+10}*4,{8+8+8+8}
		// [UNPACK(10)],[UNPACK(8)]
	}
}

INLINE void PACK(18)(const void* ints, void* bits) {
	START_PACK(18);
	for (int p = 0; p < VSCALE; p++) {
	}
}

INLINE void UNPACK(19)(const void* bits, void* ints) {
	START_UNPACK(19);
	for (int p = 0; p < VSCALE; p++) {
		// {19+13}*4,{19+6+7}*4,{19+12+1}*4,{18+3+11}*4,{16+16}*2+{8+8+8+8}
		// [UNPACK(11)],[UNPACK(8)]
	}
}

INLINE void PACK(19)(const void* ints, void* bits) {
	START_PACK(19);
	for (int p = 0; p < VSCALE; p++) {
	}
}

INLINE void UNPACK(20)(const void* bits, void* ints) {
	START_UNPACK(20);
	for (int p = 0; p < VSCALE; p++) {
		// {20+12}*4,{8+8+8+8}
		// {20+12}*2,{8+20+4}*2,{16+16}
	}
}

INLINE void PACK(20)(const void* ints, void* bits) {
	START_PACK(20);
	for (int p = 0; p < VSCALE; p++) {
	}
}

INLINE void UNPACK(21)(const void* bits, void* ints) {
	START_UNPACK(21);
	for (int p = 0; p < VSCALE; p++) {
		// {}*4,{}*4,{}*4,{}*4,{}*4,{8+8+8+8}

	}
}

INLINE void PACK(21)(const void* ints, void* bits) {
	START_PACK(21);
	for (int p = 0; p < VSCALE; p++) {
	}
}

INLINE void UNPACK(22)(const void* bits, void* ints) {
	START_UNPACK(22);
	for (int p = 0; p < VSCALE; p++) {
		// {22+10}*4,{12+6+14}*4,{16+16}*2,{8+8+8+8}
	}
}

INLINE void PACK(22)(const void* ints, void* bits) {
	START_PACK(22);
	for (int p = 0; p < VSCALE; p++) {
	}
}

INLINE void UNPACK(23)(const void* bits, void* ints) {
	START_UNPACK(23);
	for (int p = 0; p < VSCALE; p++) {
	}
}

INLINE void PACK(23)(const void* ints, void* bits) {
	START_PACK(23);
	for (int p = 0; p < VSCALE; p++) {
	}
}

INLINE void UNPACK(24)(const void* bits, void* ints) {
	START_UNPACK(24);
	for (int p = 0; p < VSCALE; p++) {
		LD_MASK(24);
		LD_MASK(8);
		LD_MASK(16);
		VECTOR xmask8_8 = SRLI32(mask8, 8);
		VECTOR xmask8_16 = SRLI32(mask8, 16);

		// 145
#if 1
		//32 sec
		VECTOR iv1 = B(0);
		VECTOR iv2 = B(1);
		VECTOR iv3 = B(2);
		I(0) = AND(iv1, mask24);
		I(1) = AND(iv2, mask24);
		I(2) = AND(iv3, mask24);
		I(3) = XOR(XOR(SRLI32(iv1, 24), SRLI32(iv2, 16)), SRLI32(iv3, 8));
#else
		//33.5sec
		VECTOR iv1 = ip[i + STRIDE * 0];
		VECTOR iv2 = ip[i + STRIDE * 1];
		VECTOR iv3 = ip[i + STRIDE * 2];
		op[i + STRIDE * 0] = AND(iv1, mask24);
		op[i + STRIDE * 1] = AND(iv2, mask24);
		op[i + STRIDE * 2] = XOR(VEC(srli_epi32)(iv1, 8), AND(iv3, mask16));
		op[i + STRIDE * 3] = XOR(VEC(srli_epi32)(iv2, 8), VEC(srli_epi32)(iv3, 16));
#endif
	}
}

INLINE void PACK(24)(const void* ints, void* bits) {
	START_PACK(24);
	for (int p = 0; p < VSCALE; p++) {
	}
}

INLINE void UNPACK(25)(const void* bits, void* ints) {
	START_UNPACK(25);
}

INLINE void PACK(25)(const void* ints, void* bits) {
	START_PACK(25);
}

INLINE void UNPACK(26)(const void* bits, void* ints) {
	START_UNPACK(26);
}

INLINE void PACK(26)(const void* ints, void* bits) {
	START_PACK(26);
}

INLINE void UNPACK(27)(const void* bits, void* ints) {
	START_UNPACK(27);
}

INLINE void PACK(28)(const void* ints, void* bits) {
	START_PACK(28);
}

INLINE void UNPACK(29)(const void* bits, void* ints) {
	START_UNPACK(29);
}

INLINE void PACK(29)(const void* ints, void* bits) {
	START_PACK(29);
}

INLINE void UNPACK(30)(const void* bits, void* ints) {
	START_UNPACK(30);
}

INLINE void PACK(30)(const void* ints, void* bits) {
	START_PACK(30);
}

INLINE void UNPACK(31)(const void* bits, void* ints) {
	START_UNPACK(31);
}

INLINE void PACK(31)(const void* ints, void* bits) {
	START_PACK(31);
}
