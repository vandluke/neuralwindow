/**@file datatype.h
 * @brief Provides common datatypes and utilities that interact with them.
 *
 */

#ifndef DATATYPE_H
#define DATATYPE_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>

typedef float float32_t;
typedef double float64_t;
typedef bool bool_t;
typedef const char * string_t;
typedef char char_t;

typedef enum datatype_t
{
    FLOAT32,
    FLOAT64
} datatype_t;

#define DATATYPES 2

string_t datatype_string(datatype_t datatype);
size_t datatype_size(datatype_t datatype);
string_t string_create(string_t format, ...);
void string_destroy(string_t string);
string_t uint64_array_to_string(const uint64_t *array, uint64_t length);

#endif
