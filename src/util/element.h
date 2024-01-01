/**@file element.h
 * @brief Provides an element of a collection and its utilities.
 *
 */

#ifndef ELEMENT_H
#define ELEMENT_H

#include <errors.h>

typedef struct element_t
{
    void *data;
    struct element_t *next;
} element_t;

typedef struct d_element_t
{
    void *data;
    struct d_element_t *next;
    struct d_element_t *prev;
} d_element_t;

nw_error_t *element_create(element_t **element, void *data);
void element_destroy(element_t *element);
nw_error_t *d_element_create(d_element_t **element, void *data);
void d_element_destroy(d_element_t *element);

#endif
