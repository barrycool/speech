#include <stdio.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
#include <errno.h>
#include <stdint.h>
#include "ringbuf.h"
#include <fcntl.h>
#include <unistd.h>

#define SHM_BUF_SIZE (96 * 1024)

int main()
{
	int shmid = shmget(ftok("/bin/bash", 0), SHM_BUF_SIZE, IPC_CREAT | 0644);

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

	ringbuf_t audio_data =  ringbuf_get(buf);

	uint8_t test_buf[3200];
	size_t test_buf_len;
	int fd = open("test_t.txt", O_CREAT | O_WRONLY, 0666);

	printf("%p\n", audio_data);
	printf("%p\n", audio_data->buf);
	printf("%lu\n", audio_data->head);
	printf("%lu\n", audio_data->tail);
	printf("%lX\n", audio_data->size);

	while (ringbuf_bytes_used(audio_data) < 3200)
	{
		usleep(10000);
	}

	for (int i = 0; i < 512; i++)
	{
		if (ringbuf_bytes_used(audio_data) >= 3200)
		{
			/*ringbuf_memcpy_from(test_buf, audio_data, 3200);*/
			ringbuf_copy_S16_S16(test_buf, audio_data, 3200);
			write(fd, test_buf, 3200);
		}
		else
		{
			i--;
			usleep(1000);
		}
	}

	close(fd);

	if (shmdt(buf) == -1)
    {
		printf("%s\n", strerror(errno));
		return 3;
    }

	return 0;
}

