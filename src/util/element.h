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

nw_error_t *element_create(element_t **element, void *data);
void element_destroy(element_t *element);

#endif
