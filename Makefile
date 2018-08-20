
all : record recognize arecord

record: record.c ringbuf.c
	gcc -g -o $@ $^

recognize: recognize.c ringbuf.c
	gcc -g -o $@ $^

arecord: aplay.c
	gcc -g -o $@ $^ -l asound

