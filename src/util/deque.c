/**@file deque.c
 * @brief Implements deque utilities.
 *
 */

#include <deque.h>

nw_error_t *deque_create(deque_t **deque)
{
    CHECK_NULL_ARGUMENT(deque, "deque");

    size_t size = sizeof(deque_t);
    *deque = (deque_t *) malloc(size);
    if (!*deque)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate deque of size %zu bytes.", size), NULL);
    }

    //Initialize
    (*deque)->head = NULL;
    (*deque)->tail = NULL;
    (*deque)->size = 0;

    return NULL;
}

void deque_destroy(deque_t *deque)
{
    if (deque)
    {
        d_element_t *element = deque->head;
        while (element)
        {
            d_element_t *next = element->next;
            d_element_destroy(element);
            element = next;
        }
    }
    free(deque);
}

nw_error_t *deque_push_front(deque_t *deque, void *data)
{
    CHECK_NULL_ARGUMENT(deque, "deque");

    d_element_t *element;
    nw_error_t *error = d_element_create(&element, data);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create element."), error);
    }

    if (!deque->head)
    {
        deque->head = element;
    }
    else
    {
        element->next = deque->head;
        deque->head->prev = element;
        deque->head = element;
    }
    ++deque->size;

    return NULL;
}

nw_error_t *deque_pop_front(deque_t *deque, void **data)
{
    CHECK_NULL_ARGUMENT(deque, "deque");
    CHECK_NULL_ARGUMENT(data, "data");

    if (!deque->head)
    {
        return ERROR(ERROR_DESTROY, string_create("failed to pop element from empty deque."), NULL);
    }

    *data = deque->head->data;
    d_element_t *element = deque->head->next;
    d_element_destroy(deque->head);
    deque->head = element;
    deque->head->prev = NULL;
    deque->size--;

    return NULL;
}

nw_error_t *deque_push_back(deque_t *deque, void *data)
{
    CHECK_NULL_ARGUMENT(deque, "deque");

    d_element_t *element;
    nw_error_t *error = d_element_create(&element, data);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create element."), error);
    }

    if (!deque->tail)
    {
        deque->tail = element;
    }
    else
    {
        element->prev = deque->tail;
        deque->tail->next = element;
        deque->tail = element;
    }
    ++deque->size;

    return NULL;
}

nw_error_t *deque_pop_back(deque_t *deque, void **data)
{
    CHECK_NULL_ARGUMENT(deque, "deque");
    CHECK_NULL_ARGUMENT(data, "data");

    if (!deque->tail)
    {
        return ERROR(ERROR_DESTROY, string_create("failed to pop element from empty deque."), NULL);
    }

    *data = deque->tail->data;
    d_element_t *element = deque->tail->prev;
    d_element_destroy(deque->tail);
    deque->tail = element;
    deque->tail->next = NULL;
    deque->size--;

    return NULL;
}
