#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <unistd.h>
#include "ringbuf.h"
#include <time.h>

#define SHM_BUF_SIZE (48 * 1024)

int main()
{
	int shmid = shmget(ftok("/record2recognize", 0), SHM_BUF_SIZE, IPC_CREAT | 0664);
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

	printf("%p\n", buf);

	ringbuf_t audio_data = ringbuf_init(buf, SHM_BUF_SIZE);

	printf("%p\n", audio_data);
	printf("%p\n", audio_data->buf);
	printf("%p\n", audio_data->head);
	printf("%ld\n", audio_data->size);

	uint8_t test_buf[1024];
	size_t test_buf_len;

	while(1)
	{
		test_buf_len = sprintf(test_buf, "%08ld\n", time(NULL));
		fprintf(stderr, test_buf);
		ringbuf_memcpy_into(audio_data, test_buf, test_buf_len);
		printf("%p\n", audio_data->buf);
		printf("%p\n", audio_data->head);
		printf("%ld\n", audio_data->size);
		sleep(1);
	}

	if (shmdt(buf) == -1)
    {
		printf("%s\n", strerror(errno));
		return 3;
    }

	if (shmctl(shmid, IPC_RMID, 0) == -1)
    {
		printf("%s\n", strerror(errno));
		return 4;
    }

	return 0;
}

