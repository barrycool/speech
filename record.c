#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include <unistd.h>
#include "ringbuf.h"
#include <time.h>
#include <fcntl.h>

#define SHM_BUF_SIZE (96 * 1024)

int main()
{
	int shmid = shmget(ftok("/bin/bash", 0), SHM_BUF_SIZE, IPC_CREAT | 0666);
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

	ringbuf_t audio_data = ringbuf_init(buf, SHM_BUF_SIZE, 4000);

	char test_buf[4000];
	size_t test_buf_len;

	int fd  = open("test.txt", O_RDONLY);

	while(1)
	{
		test_buf_len = read(fd, ringbuf_head(audio_data), 4000);
		ringbuf_fill_buf(audio_data, test_buf_len);
		if (test_buf_len < 4000)
			break;
		usleep(10000);
	}

	printf("over\n");
	close(fd);

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

