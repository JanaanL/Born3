#ifndef BORN_PROFILER_H
#define BORN_PROFILER_H

#include <iostream>
#include "papi.h"

class born_profiler_t
{
public:
	born_profiler_t ();
	virtual ~born_profiler_t ();

	void start();
	void read();
	void stop();
	void clear();
			
public:           			
	int PAPI_events[5] = { PAPI_L1_TCM , PAPI_L2_TCM, PAPI_L3_TCM, PAPI_VEC_SP, PAPI_SP_OPS };
	long long counters[5];
		
};                			

extern long total_bytes;

#endif /* end of include guard: SORT_PROFILER_H_1CK8Y26 */
