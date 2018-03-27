
#include "stdafx.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include <windows.h>

#define BLOCKCOUNT	256
#define STRIDE		256

#define INLINE inline
#include "ssedef.h"
#include "bitpack.h"

#define sleep(sec)	Sleep(sec * 1000)

const Pack packs[] = {
	PACK(1), PACK(2), PACK(3), PACK(4),
	PACK(5), PACK(6), PACK(7), PACK(8),
	PACK(9), PACK(10), PACK(11), PACK(12),
	PACK(13), PACK(14), PACK(15), PACK(16),
	PACK(17), PACK(18), PACK(19), PACK(20),
	PACK(21), PACK(22), PACK(23), PACK(24),
};

const Unpack unpacks[] = {
	UNPACK(1), UNPACK(2), UNPACK(3), UNPACK(4),
	UNPACK(5), UNPACK(6), UNPACK(7), UNPACK(8),
	UNPACK(9), UNPACK(10), UNPACK(11), UNPACK(12),
	UNPACK(13), UNPACK(14), UNPACK(15), UNPACK(16),
	UNPACK(17), UNPACK(18), UNPACK(19), UNPACK(20),
	UNPACK(21), UNPACK(22), UNPACK(23), UNPACK(24),
};

const int minimums[] = {
	BLOCKCOUNT, BLOCKCOUNT / 2, BLOCKCOUNT, BLOCKCOUNT / 4,
	BLOCKCOUNT, BLOCKCOUNT / 2, BLOCKCOUNT, BLOCKCOUNT / 8,
	BLOCKCOUNT, BLOCKCOUNT / 2, BLOCKCOUNT, BLOCKCOUNT / 4,
	BLOCKCOUNT, BLOCKCOUNT / 2, BLOCKCOUNT, BLOCKCOUNT / 8,
	BLOCKCOUNT, BLOCKCOUNT / 2, BLOCKCOUNT, BLOCKCOUNT / 4,
	BLOCKCOUNT, BLOCKCOUNT / 2, BLOCKCOUNT, BLOCKCOUNT / 8,
};
const int widths[] = {
	8, 8, 8, 8,
	8, 8, 8, 16,
	16, 16, 16, 16
};

static void* input;
static void* output;


void setup()
{
	char* in = (char*)input;
	for (int i = 0; i < 4096; i++) {
		in[i] = rand() & 0xFF;
	}
}

void verify()
{
	void* sample = malloc(4096);
	for (int i = 0; i < 11; i++) {
		const void* in = input;
		void* s = sample;
		unpacks[i](in, output);
		packs[i](output, s);
		if (memcmp(input, sample, minimums[i] / 8) == 0) {
			printf("%d] ok!\n", i);
		} else {
			printf("%d] ng!\n", i);
			unsigned short* original = (unsigned short*)output;
			unpacks[i](sample, input);
			unsigned short* rec = (unsigned short*)input;
			for (int i = 0; i < BLOCKCOUNT; i++) {
				printf("%x,", original[i]);
				if (i % 16 == 15) {
					printf("\n");
				}
			}
			printf("\n");
			for (int i = 0; i < BLOCKCOUNT; i++) {
				printf("%x,", rec[i]);
				if (i % 16 == 15) {
					printf("\n");
				}
			}
			printf("\n");
			sleep(120);
		}
	}
}

void speed()
{
	for (int k = 0; k < 5; k++) {
		UNPACK(9)(input, output);
		clock_t begin = clock();
		for (int i = 0; i < 10000 * 10000 * 10; i++) {
			for (int v = 0; v < 1/*BLOCKCOUNT / minimums[11]*/; v++) {
				UNPACK(9)(input, output);
			}
		}
		clock_t end = clock();
		printf("%f\n", (double)(end - begin) / CLOCKS_PER_SEC);
		sleep(1);
	}
}



int main()
{
	input = malloc(4096);
	output = malloc(4096);
	setup();
	if (1) {
		verify();
	} else {
		sleep(10);
		speed();
	}
	sleep(120);
}