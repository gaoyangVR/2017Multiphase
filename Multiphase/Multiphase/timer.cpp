#ifndef _TIMER
#define _TIMER
#include <windows.h>
#include "timer.h"

double CTimer::LIToSecs(LARGE_INTEGER & L)
{
	return ((double)L.QuadPart / (double)frequency.QuadPart);
}

double CTimer::LIToMS(LARGE_INTEGER & L)
{
	return ((double)L.QuadPart / (double)frequency.QuadPart * 1000);
}

CTimer::CTimer()
{
	mTimer.start.QuadPart = 0;
	mTimer.stop.QuadPart = 0;
	QueryPerformanceFrequency(&frequency);
	QueryPerformanceCounter(&mTimer.start);
}

void CTimer::startTimer()
{
	QueryPerformanceCounter(&mTimer.start);
}

void CTimer::stopTimer()
{
	QueryPerformanceCounter(&mTimer.stop);
}

double CTimer::getElapsedTime()
{
	LARGE_INTEGER time;
	time.QuadPart = mTimer.stop.QuadPart - mTimer.start.QuadPart;
	return LIToSecs(time);
}

double CTimer::stopgetstartS()
{
	QueryPerformanceCounter(&mTimer.stop);
	LARGE_INTEGER time;
	time.QuadPart = mTimer.stop.QuadPart - mTimer.start.QuadPart;
	QueryPerformanceCounter(&mTimer.start);
	return LIToSecs(time);
}

double CTimer::stopgetstartMS()
{
	QueryPerformanceCounter(&mTimer.stop);
	LARGE_INTEGER time;
	time.QuadPart = mTimer.stop.QuadPart - mTimer.start.QuadPart;
	QueryPerformanceCounter(&mTimer.start);
	return LIToMS(time);
}

#endif
