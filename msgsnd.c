#include <stdio.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <unistd.h>

#define MSG_TYPE_NEW_TXT 1

#define	MSG_LEN 32

struct msgbuf {
	long mtype;
	char mtext[MSG_LEN];
};

int main(int argc, char **argv)
{

	if (argc < 2)
	{
		printf("wrong parameter\n");
		return 1;
	}

	int msgId = msgget(ftok("/bin/ps", 0), IPC_CREAT | 0666);
	if (msgId == -1) {
		perror("msgget");
		return 2;
	}

	struct msgbuf msg;

	msg.mtype = MSG_TYPE_NEW_TXT;
	snprintf(msg.mtext, MSG_LEN, "%s", argv[1]);

	if (msgsnd(msgId, (void *) &msg, MSG_LEN, IPC_NOWAIT) == -1) {
		perror("msgsnd error");
		return 3;
	}


	return 0;
}

