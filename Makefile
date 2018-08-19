
all : record recognize


record: record.c ringbuf.c
	gcc -g -o $@ $^

recognize: recognize.c ringbuf.c
	gcc -g -o $@ $^

