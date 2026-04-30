#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <fcntl.h>
#include <string>
#include <chrono>
#include <fstream>
#include <limits>
#include <random>
#include <iostream>
#include <sstream>
#include <thread>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <mqueue.h>

using namespace std;

#define MQ_BUFFER_SIZE 8192

#define BUG_ON(cond)											\
	do {												\
		if (cond) {										\
			fprintf(stderr, "BUG_ON: %s (L%d) %s\n", __FILE__, __LINE__, __FUNCTION__);	\
			raise(SIGABRT);									\
		}											\
	} while (0)


int main(int argc, char *argv[]) {
	if (argc != 4) {
		printf("Usage: %s <request queue name> <response queue name> <message>\n", argv[0]);
		exit(1);
	}
	char *request_queue_name = argv[1];
	char *response_queue_name = argv[2];
	string message(argv[3]);

	mqd_t request_queue = mq_open(request_queue_name, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH, NULL);
	BUG_ON(request_queue == (mqd_t) -1);

	mqd_t response_queue = mq_open(response_queue_name, O_RDONLY | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH, NULL);
	BUG_ON(response_queue == (mqd_t) -1);

	BUG_ON(mq_send(request_queue, message.c_str(), message.size() + 1, 0) == -1);

	char buffer[MQ_BUFFER_SIZE];
	BUG_ON(mq_receive(response_queue, buffer, MQ_BUFFER_SIZE, NULL) == -1);
	return 0;
}
