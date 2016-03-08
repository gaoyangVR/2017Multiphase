#ifndef _STOP_TIMER
#define _STOP_TIMER
#include <windows.h>

typedef struct
{
	LARGE_INTEGER start;
	LARGE_INTEGER stop;
} timer;

class CTimer
{

private:
	timer mTimer;
	LARGE_INTEGER frequency;
	double LIToSecs(LARGE_INTEGER & L);
	double LIToMS(LARGE_INTEGER & L);
public:
	CTimer();
	void startTimer();
	void stopTimer();
	double getElapsedTime();
	double stopgetstartS();
	double stopgetstartMS();
};
#endif