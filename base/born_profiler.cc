#include "born_profiler.h"
#include "papi.h"
#include <iostream>

long                    total_bytes;

born_profiler_t::born_profiler_t () {
//	p_flpops =   0;   // papi floating point operations
//	_pri_p_flpops =   0;
	int retval;
	retval = PAPI_library_init(PAPI_VER_CURRENT);
	if (retval != PAPI_VER_CURRENT)
		std::cout << "PAPI library init error!" << std::endl;

	if (PAPI_query_event(PAPI_L1_TCM) != PAPI_OK)
		std::cout << "PAPI could not find the PAPI_L1_TCM event" << std::endl;
//	else
	//	std::cout << "PAPI found PAPI_L1_TCM event " << std::endl;
	
	if (PAPI_query_event(PAPI_L2_TCM) != PAPI_OK)
		std::cout << "PAPI could not find the PAPI_L2_TCM event" << std::endl;
//	else
	//	std::cout << "PAPI found PAPI_L1_DCM event " << std::endl;
	
	if (PAPI_query_event(PAPI_L3_TCM) != PAPI_OK)
		std::cout << "PAPI could not find the PAPI_L3_TCM event" << std::endl;
//	else
	//	std::cout << "PAPI found PAPI_L2_TCM event " << std::endl;
	
	
	if (PAPI_query_event(PAPI_SP_OPS) != PAPI_OK)
		std::cout << "PAPI could not find the PAPI_SP_OPS event" << std::endl;
//	else
	//	std::cout << "PAPI found PAPI_L2_DCM event " << std::endl;
	
	if (PAPI_query_event(PAPI_VEC_SP) != PAPI_OK)
		std::cout << "PAPI could not find the PAPI_VEC_SP event" << std::endl;
//	else
	//	std::cout << "PAPI found PAPI_L2_DCM event " << std::endl;
	
}

born_profiler_t::~born_profiler_t () {
	
}

void
born_profiler_t::start() {
	int ret;
	if ((ret = PAPI_start_counters(PAPI_events, 3)) != PAPI_OK){
		std::cout << "PAPI failed to start counters: " << ret << std::endl;
	}
}

void
born_profiler_t::read() {
	int ret;
	if((ret = PAPI_read_counters(counters, 3))!= PAPI_OK){
		std::cout << "PAPI failed to read counters: " << ret << std::endl;
	}
	std::cout << "L1 data cache misses is " << counters[0] << std::endl;
	std::cout << "L2 data cache misses is " << counters[1] << std::endl;
	std::cout << "L3 data cache misses is " << counters[2] << std::endl;
//	std::cout << "Total FLOPS is  " << counters[3] << std::endl;
//	std::cout << "Total vector instructions are  " << counters[4] << std::endl;
}
void
born_profiler_t::stop() {
	int ret;
	if((ret = PAPI_stop_counters(counters, 3))!= PAPI_OK){
		std::cout << "PAPI failed to stop counters: " << ret << std::endl;
//	std::cout << "Total hardware flops = " << (float) counters[1] << std::endl;
	}
}

void
born_profiler_t::clear() {
}

//void 
//born_profiler_t::flops_papi() {
//	int 		retval;
//	float rtime, ptime, mflops;
//	retval  = PAPI_flops(&rtime, &ptime, &_pri_p_flpops, &mflops);
	//assert (retval == PAPI_OK);
//	std::cout << "Real time: " << rtime << " Process Time: " << ptime << " Total FLOPS: " << _pri_p_flpops << "MFLOPS: " << mflops << std::endl;
//	_pri_p_flpops =   0;
//}
