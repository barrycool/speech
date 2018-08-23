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

int main()
{
	int msgId = msgget(ftok("/bin/ps", 0), IPC_CREAT | 0666);
	if (msgId == -1) {
		perror("msgget");
		return 2;
	}

	struct msgbuf msg;

	if (msgrcv(msgId, (void *) &msg, MSG_LEN, MSG_TYPE_NEW_TXT, 0 /*IPC_NOWAIT*/) == -1) {
		perror("msgsnd error");
		return 3;
	}

	printf("%s\n", msg.mtext);

	return 0;
}

