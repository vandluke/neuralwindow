/**@file deque.h
 * @brief Provides deque type and its utilities.
 *
 */

#ifndef DEQUE_H
#define DEQUE_H

#include <element.h>
#include <datatype.h>
#include <errors.h>

typedef struct deque_t
{
    d_element_t *head;
    d_element_t *tail;
    uint64_t size;
} deque_t;

nw_error_t *deque_create(deque_t **deque);
void deque_destroy(deque_t *deque);
nw_error_t *deque_push_front(deque_t *deque, void *data);
nw_error_t *deque_pop_front(deque_t *deque, void **data);
nw_error_t *deque_push_back(deque_t *deque, void *data);
nw_error_t *deque_pop_back(deque_t *deque, void **data);

#endif
