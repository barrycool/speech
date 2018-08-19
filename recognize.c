#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <unistd.h>
#include "ringbuf.h"

#define SHM_BUF_SIZE (48 * 1024)

int main()
{
	int shmid = shmget(ftok("/record2recognize", 0), SHM_BUF_SIZE, IPC_CREAT | 0644);

	if (shmid == -1)
	{
		printf("%s\n", strerror(errno));
		return 1;
	}

	uint8_t *buf = shmat(shmid, NULL, 0);
	if (buf == NULL)
	{
		printf("%s\n", strerror(errno));
		return 2;
	}

	ringbuf_t audio_data =  ringbuf_init(buf, SHM_BUF_SIZE);

	uint8_t test_buf[1024];
	size_t test_buf_len;

	while(1)
	{
		ringbuf_memcpy_from(test_buf, audio_data, 9);

		printf("%s\n", buf);
		
		sleep(1);
	}

	if (shmdt(buf) == -1)
    {
		printf("%s\n", strerror(errno));
		return 3;
    }

	return 0;
}

