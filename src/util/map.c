/**@file map.c
 * @brief Implements hash-map utilities.
 *
 */

#include <map.h>
#include <string.h>

// djb2
static uint64_t map_hash_key(string_t string)
{
    uint64_t hash = 5381;
    int c;

    while ((c = *string++))
    {
        hash = ((hash << 5) + hash) + c;
    }

    return hash;
}

nw_error_t *map_create(map_t **map)
{
    CHECK_NULL_ARGUMENT(map, "map");

    size_t size = sizeof(map_t);
    *map = (map_t *) malloc(size);
    if (!*map)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate map of size %zu bytes.", size),
                     NULL);
    }
    (*map)->length = 0;
    (*map)->capacity = INITIAL_CAPACITY;
    size = (*map)->capacity * sizeof(entry_t);
    (*map)->entries = (entry_t *) malloc(size);
    if (!(*map)->entries)
    {
        free(*map);
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate map->entries of size %zu bytes.", size),
                     NULL);
    }

    for (uint64_t i = 0; i < (*map)->capacity; ++i)
    {
        (*map)->entries[i].key = NULL;
        (*map)->entries[i].data = NULL;
    }

    return NULL;
}

void map_destroy(map_t *map)
{
    if (map)
    {
        if (map->entries)
        {
            for (uint64_t i = 0; i < map->capacity; ++i)
            {
                string_destroy(map->entries[i].key);
            }
        }
        free(map->entries);
    }
    free(map);
}

nw_error_t *map_get(map_t *map, string_t key, void **data)
{
    CHECK_NULL_ARGUMENT(map, "map");
    CHECK_NULL_ARGUMENT(map->entries, "map->entries");
    CHECK_NULL_ARGUMENT(key, "key");
    CHECK_NULL_ARGUMENT(data, "data");

    uint64_t hash = map_hash_key(key);
    uint64_t index = (uint64_t)(hash & (uint64_t)(map->capacity - 1));
    while (map->entries[index].key)
    {
        if (strcmp(key, map->entries[index].key) == 0)
        {
            (*data) = map->entries[index].data;
            return NULL;
        }
        ++index;
        if (index >= map->capacity)
        {
            index = 0;
        }
    }
    return NULL;
}

static nw_error_t *map_set_entry(entry_t *entries, uint64_t capacity, string_t key, void *data, uint64_t *length)
{
    CHECK_NULL_ARGUMENT(entries, "entries");
    CHECK_NULL_ARGUMENT(key, "key");

    uint64_t hash = map_hash_key(key);
    uint64_t index = (uint64_t)(hash & (uint64_t)(capacity - 1));
    while (entries[index].key)
    {
        if (strcmp(key, entries[index].key) == 0)
        {
            entries[index].data = data;
            free((char *) key);
            return NULL;
        }
        ++index;
        if (index >= capacity)
        {
            index = 0;
        }
    }

    entries[index].key = key;
    entries[index].data = data;
    ++(*length);

    return NULL;
}

bool_t map_contains(map_t *map, string_t key)
{
    if (!map || !map->entries || !key)
    {
        return false;
    }

    uint64_t hash = map_hash_key(key);
    uint64_t index = (uint64_t)(hash & (uint64_t)(map->capacity - 1));
    while (map->entries[index].key)
    {
        if (strcmp(key, map->entries[index].key) == 0)
        {
            return true;
        }
        ++index;
        if (index >= map->capacity)
        {
            index = 0;
        }
    }
    return false;
}

static nw_error_t *map_expand(map_t *map)
{
    CHECK_NULL_ARGUMENT(map, "map");
    CHECK_NULL_ARGUMENT(map->entries, "map->entries");

    uint64_t new_length = 0;
    uint64_t new_capacity = map->capacity * 2;
    if (new_capacity < map->capacity)
    {
        return ERROR(ERROR_OVERFLOW,
                     string_create("capacity %lu is too large to be doubled.",
                     map->capacity), NULL);
    }

    size_t size = new_capacity * sizeof(entry_t);
    entry_t *new_entries = (entry_t *) malloc(size);
    if (!new_entries)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION,
                     string_create("failed to allocate map entries of size %zu bytes.", size),
                     NULL);
    }
    for (uint64_t i = 0; i < new_capacity; ++i)
    {
        new_entries[i].key = NULL;
        new_entries[i].data = NULL;
    }
    
    for (uint64_t i = 0; i < map->capacity; ++i)
    {
        entry_t entry = map->entries[i];
        if (entry.key)
        {
            nw_error_t *error = map_set_entry(new_entries, new_capacity, entry.key, entry.data, &new_length);
            if (error)
            {
                free(new_entries);
                return ERROR(ERROR_SET, string_create("failed to set entry with corresponding key %s.", entry.key), NULL);
            }
        }
    }

    free(map->entries);
    map->entries = new_entries;
    map->capacity = new_capacity;
    map->length = new_length;

    return NULL;
}

nw_error_t *map_set(map_t *map, string_t key, void *data)
{
    CHECK_NULL_ARGUMENT(map, "map");
    CHECK_NULL_ARGUMENT(map->entries, "map->entries");
    CHECK_NULL_ARGUMENT(key, "key");

    nw_error_t *error;
    if (map->length >= map->capacity / 2)
    {
        error = map_expand(map);
        if (error)
        {
            return ERROR(ERROR_EXPAND, string_create("failed to increase capacity of map."), error);
        }
    }

    char *copy_key = (char *) malloc(sizeof(char) * (strlen(key) + 1));
    if (!copy_key)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate memory."), NULL);
    }

    strcpy(copy_key, key);

    error = map_set_entry(map->entries, map->capacity, copy_key, data, &map->length);
    if (error)
    {
        return ERROR(ERROR_SET, string_create("failed to set map entry with corresponding key %s.", key), error);
    }

    return NULL;
}
