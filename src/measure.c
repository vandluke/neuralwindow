#include <measure.h>
#include <time.h>

uint64_t get_time_ns(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (uint64_t) (ts.tv_sec * 1e9) + (uint64_t) ts.tv_nsec;
}