#ifndef __TIMER_H__INCLUDED__
#define __TIMER_H__INCLUDED__

#define _CRTDBG_MAP_ALLOC_ // added for debug
#include <stdlib.h>
#include <crtdbg.h> // added for debug

#include <ctime>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdio.h>

#include <string.h>




class Timer {
	std::clock_t tS;
	std::clock_t finish;
	bool outPut;
	bool run;
	size_t size;
	std::ofstream file;
	std::string fName;
	std::string save;
	void write();
public:
	Timer(std::string);
	~Timer();
	size_t maxSize = 500 * 1000;
	void lap(std::string);
};
//Define constructor
Timer::Timer(std::string f = "") {
	outPut = false;
	run = true;
	if (f == "false") {
		run = false;
	}
	else if (f != "") {
		outPut = true;
		file.open(fName);
		file << "Event, Elapsed time (s)" << std::endl;
	}
	tS = std::clock();
	//finish = time(NULL);
	lap("Start:");
}
Timer::~Timer() {
	if (outPut) {
		write();
	}
}
//Define Destructor
void Timer::write() {
	file << save << std::endl;
	save.clear();
}
void Timer::lap(std::string prompt = "") {
	finish = std::clock();
	double dt = (long double)(finish - tS) / ((double)CLOCKS_PER_SEC);
	if (run) {
		printf("%s: %g\n", prompt.c_str(), dt);
		if (outPut) {
			save.append(prompt);
			save.append(",");
			save.append(std::to_string(dt));
			save.append("\n");
			if (save.length() > maxSize) {
				write();
			}
		}
	};
	return;
}
#endif // !__TIMER_H__INCLUDED__