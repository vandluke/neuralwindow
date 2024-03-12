#include <layer.h>
#include <init.h>
#include <view.h>
#include <buffer.h>
#include <tensor.h>
#include <function.h>
#include <math.h>
#include <string.h>
#include <map.h>

extern bool_t no_gradient;

nw_error_t *model_create(model_t **model, block_t *block)
{
    CHECK_NULL_ARGUMENT(model, "model");
    CHECK_NULL_ARGUMENT(block, "block");

    *model = (model_t *) malloc(sizeof(model_t));
    if (!*model)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(model_t)), NULL);
    }

    (*model)->block = block;

    return NULL;
}

void model_destroy(model_t *model) 
{
    if (model)
    {
        block_destroy(model->block);
        free(model);
    }
}

nw_error_t *block_create(block_t **block, int64_t depth, ...)
{
    CHECK_NULL_ARGUMENT(block, "block");
    
    va_list valist;
    va_start(valist, depth);

    *block = (block_t *) malloc(sizeof(block_t));
    if (!*block)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(block_t)), NULL);
    }

    (*block)->depth = depth;
    (*block)->layers = (layer_t **) malloc(depth * sizeof(layer_t *));
    if (!(*block)->layers)
    {
        free(*block);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(depth * sizeof(layer_t *))), NULL);
    }

    for (int64_t i = 0; i < depth; ++i)
    {
        (*block)->layers[i] = (layer_t *) va_arg(valist, layer_t *);
    }

    va_end(valist);

    return NULL;
}

nw_error_t *block_create_from_array(block_t **block, int64_t depth, layer_t **layers)
{
    CHECK_NULL_ARGUMENT(block, "block");
    CHECK_NULL_ARGUMENT(layers, "layers");

    *block = (block_t *) malloc(sizeof(block_t));
    if (!*block)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(block_t)), NULL);
    }

    (*block)->depth = depth;
    (*block)->layers = (layer_t **) malloc(depth * sizeof(layer_t *));
    if (!(*block)->layers)
    {
        free(*block);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(depth * sizeof(layer_t *))), NULL);
    }

    for (int64_t i = 0; i < depth; ++i)
    {
        (*block)->layers[i] = (layer_t *) layers[i];
    }

    return NULL;
}

void block_destroy(block_t *block)
{
    if (block)
    {
        if (block->layers)
        {
            for (int64_t i = 0; i < block->depth; ++i)
            {
                layer_destroy(block->layers[i]);
            }
            free(block->layers);
        }
        free(block);
    }
}

nw_error_t *layer_create(layer_t **layer, transform_t *transform, transform_type_t transform_type)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(transform, "transform");

    *layer = (layer_t *) malloc(sizeof(layer_t));
    if (!*layer)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(layer_t)), NULL);
    }

    (*layer)->transform = transform;
    (*layer)->transform_type = transform_type;

    return NULL;
}

void layer_destroy(layer_t *layer)
{
    if (layer)
    {
        transform_destroy(layer->transform, layer->transform_type);
        free(layer);
    }
}

nw_error_t *transform_create(transform_t **transform, transform_type_t transform_type, void *type_transform)
{
    CHECK_NULL_ARGUMENT(transform, "transform");
    CHECK_NULL_ARGUMENT(type_transform, "type_transform");

    *transform = (transform_t *) malloc(sizeof(transform_t));
    if (!*transform)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(transform_t)), NULL);
    }

    switch (transform_type)
    {
    case LINEAR:
        (*transform)->linear = (linear_t *) type_transform;
        break;
    case CONVOLUTION_2D:
    case CONVOLUTION_TRANSPOSE_2D:
        (*transform)->convolution_2d = (convolution_2d_t *) type_transform;
        break;
    case MAX_POOLING_2D:
    case AVERAGE_POOLING_2D:
        (*transform)->pooling_2d = (pooling_2d_t *) type_transform;
        break;
    case DROPOUT:
        (*transform)->dropout = (dropout_t *) type_transform;
        break;
    case BATCH_NORMALIZATION_2D:
        (*transform)->batch_normalization_2d = (batch_normalization_2d_t *) type_transform;
        break;
    case LAYER_NORMALIZATION:
        (*transform)->layer_normalization = (layer_normalization_t *) type_transform;
        break;
    case RESHAPE:
        (*transform)->reshape = (reshape_t *) type_transform;
        break;
    case EMBEDDING:
        (*transform)->embedding = (embedding_t *) type_transform;
        break;
    case TRANSFORMER_EMBEDDING:
        (*transform)->transformer_embedding = (transformer_embedding_t *) type_transform;
        break;
    case CAUSAL_MULTIHEAD_SELF_ATTENTION:
        (*transform)->causal_multihead_self_attention = (causal_multihead_self_attention_t *) type_transform;
        break;
    case ACTIVATION:
        (*transform)->activation = (activation_t *) type_transform;
        break;
    case BLOCK:
    case RESIDUAL_BLOCK:
        (*transform)->block = (block_t *) type_transform;
        break;
    default:
        free(*transform);
        return ERROR(ERROR_LAYER_TYPE, string_create("unknown transform type %d.", (int) transform_type), NULL);
    }

    return NULL;
}

void transform_destroy(transform_t *transform, transform_type_t transform_type)
{
    if (transform)
    {
        switch (transform_type)
        {
        case LINEAR:
            linear_destroy(transform->linear);
            break;
        case CONVOLUTION_2D:
        case CONVOLUTION_TRANSPOSE_2D:
            convolution_2d_destroy(transform->convolution_2d);
            break;
        case MAX_POOLING_2D:
        case AVERAGE_POOLING_2D:
            pooling_2d_destroy(transform->pooling_2d);
            break;
        case DROPOUT:
            dropout_destroy(transform->dropout);
            break;
        case BATCH_NORMALIZATION_2D:
            batch_normalization_2d_destroy(transform->batch_normalization_2d);
            break;
        case LAYER_NORMALIZATION:
            layer_normalization_destroy(transform->layer_normalization);
            break;
        case RESHAPE:
            reshape_destroy(transform->reshape);
            break;
        case EMBEDDING:
            embedding_destroy(transform->embedding);
            break;
        case TRANSFORMER_EMBEDDING:
            transformer_embedding_destroy(transform->transformer_embedding);
            break;
        case CAUSAL_MULTIHEAD_SELF_ATTENTION:
            causal_multihead_self_attention_destroy(transform->causal_multihead_self_attention);
            break;
        case ACTIVATION:
            activation_destroy(transform->activation);
            break;
        case BLOCK:
        case RESIDUAL_BLOCK:
            block_destroy(transform->block);
            break;
        default:
            break;
        }
        free(transform);
    }
}

string_t transform_type_string(transform_type_t transform_type)
{
    switch (transform_type)
    {
    case LINEAR:
        return "LINEAR";
    case CONVOLUTION_2D:
        return "CONVOLUTION_2D";
    case CONVOLUTION_TRANSPOSE_2D:
        return "CONVOLUTION_TRANSPOSE_2D";
    case MAX_POOLING_2D:
        return "MAX_POOLING_2D";
    case AVERAGE_POOLING_2D:
        return "AVERAGE_POOLING_2D";
    case DROPOUT:
        return "DROPOUT";
    case BATCH_NORMALIZATION_2D:
        return "BATCH_NORMALIZATION_2D";
    case LAYER_NORMALIZATION:
        return "LAYER_NORMALIZATION";
    case RESHAPE:
        return "RESHAPE";
   case EMBEDDING:
        return "EMBEDDING";
   case TRANSFORMER_EMBEDDING:
        return "TRANSFORMER_EMBEDDING";
   case CAUSAL_MULTIHEAD_SELF_ATTENTION:
        return "CAUSAL_MULTIHEAD_SELF_ATTENTION";
    case ACTIVATION:
        return "ACTIVATION";
    case BLOCK:
        return "BLOCK";
    case RESIDUAL_BLOCK:
        return "RESIDUAL_BLOCK";
    default:
        return "TRANSFORM_TYPE";
    }
}

nw_error_t *linear_create(linear_t **linear, tensor_t *weights, tensor_t *bias)
{
    CHECK_NULL_ARGUMENT(linear, "linear");
    CHECK_NULL_ARGUMENT(weights, "weights");

    *linear = (linear_t *) malloc(sizeof(linear_t));
    if (!*linear)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(linear_t)), NULL);
    }

    (*linear)->weights = weights;
    (*linear)->bias = bias;

    return NULL;
}

void linear_destroy(linear_t *linear)
{
    if (linear)
    {
        tensor_destroy(linear->weights);
        tensor_destroy(linear->bias);
        free(linear);
    }
}

nw_error_t *convolution_2d_create(convolution_2d_t **convolution_2d, int64_t padding, int64_t stride, tensor_t *kernel, tensor_t *bias)
{
    CHECK_NULL_ARGUMENT(convolution_2d, "convolution_2d");
    CHECK_NULL_ARGUMENT(kernel, "kernel");

    *convolution_2d = (convolution_2d_t *) malloc(sizeof(convolution_2d_t));
    if (!*convolution_2d)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(convolution_2d_t)), NULL);
    }

    (*convolution_2d)->padding = padding;
    (*convolution_2d)->stride = stride;
    (*convolution_2d)->kernel = kernel;
    (*convolution_2d)->bias = bias;

    return NULL;
}

void convolution_2d_destroy(convolution_2d_t *convolution_2d)
{
    if (convolution_2d)
    {
        tensor_destroy(convolution_2d->kernel);
        tensor_destroy(convolution_2d->bias);
        free(convolution_2d);
    }
}

nw_error_t *pooling_2d_create(pooling_2d_t **pooling_2d, int64_t padding, int64_t stride, int64_t kernel)
{
    CHECK_NULL_ARGUMENT(pooling_2d, "pooling_2d");

    *pooling_2d = (pooling_2d_t *) malloc(sizeof(pooling_2d_t));
    if (!*pooling_2d)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(pooling_2d_t)), NULL);
    }

    (*pooling_2d)->padding = padding;
    (*pooling_2d)->stride = stride;
    (*pooling_2d)->kernel = kernel;

    return NULL;
}

void pooling_2d_destroy(pooling_2d_t *pooling_2d)
{
    if (pooling_2d)
    {
        free(pooling_2d);
    }
}

nw_error_t *dropout_create(dropout_t **dropout, void *probability, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(dropout, "dropout");
    CHECK_NULL_ARGUMENT(probability, "probability");

    *dropout = (dropout_t *) malloc(sizeof(dropout_t));
    if (!*dropout)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(dropout_t)), NULL);
    }

    (*dropout)->probability = (void *) malloc(datatype_size(datatype));
    if (!(*dropout)->probability)
    {
        free(*dropout);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size(datatype)), NULL);
    }

    memcpy((*dropout)->probability, probability, datatype_size(datatype));
    (*dropout)->inference = false;
    (*dropout)->datatype = datatype;

    return NULL;
}

void dropout_destroy(dropout_t *dropout)
{
    if (dropout)
    {
        free(dropout->probability);
        free(dropout);
    }
}

nw_error_t *batch_normalization_2d_create(batch_normalization_2d_t **batch_normalization_2d, int64_t number_of_features,
                                          void *momentum, void *epsilon, bool_t track_running_stats,
                                          bool_t affine, datatype_t datatype, runtime_t runtime)
{
    CHECK_NULL_ARGUMENT(batch_normalization_2d, "batch_normalization_2d");
    CHECK_NULL_ARGUMENT(epsilon, "epsilon");
    if (track_running_stats)
    {
        CHECK_NULL_ARGUMENT(momentum, "momentum");
    }

    *batch_normalization_2d = (batch_normalization_2d_t *) malloc(sizeof(batch_normalization_2d_t));
    if (!*batch_normalization_2d)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(batch_normalization_2d_t)), NULL);
    }

    nw_error_t *error = NULL;

    (*batch_normalization_2d)->momentum = NULL;
    (*batch_normalization_2d)->epsilon = NULL;
    (*batch_normalization_2d)->bias = NULL;
    (*batch_normalization_2d)->weights = NULL;
    (*batch_normalization_2d)->running_mean = NULL;
    (*batch_normalization_2d)->running_variance = NULL;
    (*batch_normalization_2d)->track_running_stats = track_running_stats;
    (*batch_normalization_2d)->inference = false;
    (*batch_normalization_2d)->datatype = datatype;

    (*batch_normalization_2d)->momentum = (void *) malloc(datatype_size(datatype));
    if (!(*batch_normalization_2d)->momentum)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size(datatype)), NULL);
        goto cleanup;
    }
    memcpy((*batch_normalization_2d)->momentum, momentum, datatype_size(datatype));

    (*batch_normalization_2d)->epsilon = (void *) malloc(datatype_size(datatype));
    if (!(*batch_normalization_2d)->epsilon)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size(datatype)), NULL);
        goto cleanup;
    }
    memcpy((*batch_normalization_2d)->epsilon, epsilon, datatype_size(datatype));

    if (affine)
    {
        error = tensor_create_ones(&(*batch_normalization_2d)->weights, (int64_t[]){number_of_features}, 1, runtime, datatype, true, true);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }

        error = tensor_create_zeroes(&(*batch_normalization_2d)->bias, (int64_t[]){number_of_features}, 1, runtime, datatype, true, true);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }
    }

    error = tensor_create_ones(&(*batch_normalization_2d)->running_variance, (int64_t[]){number_of_features}, 1, runtime, datatype, false, true);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_create_zeroes(&(*batch_normalization_2d)->running_mean, (int64_t[]){number_of_features}, 1, runtime, datatype, false, true);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }
    
    return error;

cleanup:

    batch_normalization_2d_destroy(*batch_normalization_2d);

    return error;
}

void batch_normalization_2d_destroy(batch_normalization_2d_t *batch_normalization_2d)
{
    if (batch_normalization_2d)
    {
        tensor_destroy(batch_normalization_2d->weights);
        tensor_destroy(batch_normalization_2d->bias);
        tensor_destroy(batch_normalization_2d->running_mean);
        tensor_destroy(batch_normalization_2d->running_variance);
        free(batch_normalization_2d->epsilon);
        free(batch_normalization_2d->momentum);
        free(batch_normalization_2d);
    }
}

nw_error_t *layer_normalization_create(layer_normalization_t **layer_normalization, const int64_t *normalized_shape, int64_t length,
                                        void *epsilon, bool_t bias_affine, bool_t weights_affine, datatype_t datatype, runtime_t runtime)
{
    CHECK_NULL_ARGUMENT(layer_normalization, "layer_normalization");
    CHECK_NULL_ARGUMENT(epsilon, "epsilon");
    CHECK_NULL_ARGUMENT(normalized_shape, "normalized_shape");

    *layer_normalization = (layer_normalization_t *) malloc(sizeof(layer_normalization_t));
    if (!*layer_normalization)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(layer_normalization_t)), NULL);
    }

    nw_error_t *error = NULL;

    (*layer_normalization)->epsilon = NULL;
    (*layer_normalization)->bias = NULL;
    (*layer_normalization)->weights = NULL;
    (*layer_normalization)->length = length;
    (*layer_normalization)->normalized_shape = NULL;
    (*layer_normalization)->datatype = datatype;
    
    size_t size = datatype_size(datatype);
    (*layer_normalization)->epsilon = (void *) malloc(size);
    if (!(*layer_normalization)->epsilon)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }
    memcpy((*layer_normalization)->epsilon, epsilon, size);

    size = length * sizeof(int64_t);
    (*layer_normalization)->normalized_shape = (int64_t *) malloc(size);
    if (!(*layer_normalization)->normalized_shape)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }
    memcpy((*layer_normalization)->normalized_shape, normalized_shape, size);

    if (weights_affine)
    {
        error = tensor_create_ones(&(*layer_normalization)->weights, normalized_shape, length, runtime, datatype, true, true);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }
    }

    if (bias_affine)
    {
        error = tensor_create_zeroes(&(*layer_normalization)->bias, normalized_shape, length, runtime, datatype, true, true);
        if (error)
        {
            error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
            goto cleanup;
        }
    }

    return error;

cleanup:

    layer_normalization_destroy(*layer_normalization);

    return error;
}

void layer_normalization_destroy(layer_normalization_t *layer_normalization)
{
    if (layer_normalization)
    {
        tensor_destroy(layer_normalization->weights);
        tensor_destroy(layer_normalization->bias);
        free(layer_normalization->epsilon);
        free(layer_normalization->normalized_shape);
        free(layer_normalization);
    }
}

nw_error_t *reshape_create(reshape_t **reshape, int64_t *shape, int64_t length)
{
    CHECK_NULL_ARGUMENT(reshape, "reshape");
    CHECK_NULL_ARGUMENT(shape, "shape");

    *reshape = (reshape_t *) malloc(sizeof(reshape_t));
    if (!*reshape)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(reshape_t)), NULL);
    }

    (*reshape)->shape = NULL;
    (*reshape)->length = length;

    (*reshape)->shape = (int64_t *) malloc(length * sizeof(int64_t));
    if (!(*reshape)->shape)
    {
        free(*reshape);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", length * sizeof(int64_t)), NULL);
    }

    for (int64_t i = 0; i < length; ++i)
    {
        (*reshape)->shape[i] = shape[i];
    }

    return NULL;
}

void reshape_destroy(reshape_t *reshape)
{
    if (reshape)
    {
        free(reshape->shape);
        free(reshape);
    }
}

nw_error_t *embedding_create(embedding_t **embedding, int64_t vocabulary_size, int64_t embedding_size, tensor_t *vocabulary_counter, tensor_t *weights)
{
    CHECK_NULL_ARGUMENT(embedding, "embedding");
    CHECK_NULL_ARGUMENT(vocabulary_counter, "vocabulary_counter");
    CHECK_NULL_ARGUMENT(weights, "weights");

    *embedding = (embedding_t *) malloc(sizeof(embedding_t));
    if (!*embedding)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(embedding_t)), NULL);
    }

    (*embedding)->vocabulary_size = vocabulary_size;
    (*embedding)->embedding_size = embedding_size;
    (*embedding)->vocabulary_counter = vocabulary_counter;
    (*embedding)->weights = weights;

    return NULL;
}

void embedding_destroy(embedding_t *embedding)
{
    tensor_destroy(embedding->vocabulary_counter);
    tensor_destroy(embedding->weights);
    free(embedding);
}

nw_error_t *transformer_embedding_create(transformer_embedding_t **transformer_embedding, embedding_t *token_embedding, embedding_t *position_embedding)
{
    CHECK_NULL_ARGUMENT(transformer_embedding, "transformer_embedding");
    CHECK_NULL_ARGUMENT(token_embedding, "token_embedding");
    CHECK_NULL_ARGUMENT(position_embedding, "position_embedding");

    *transformer_embedding = (transformer_embedding_t *) malloc(sizeof(transformer_embedding_t));
    if (!*transformer_embedding)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(transformer_embedding_t)), NULL);
    }

    (*transformer_embedding)->token_embedding = token_embedding;
    (*transformer_embedding)->position_embedding = position_embedding;

    return NULL;
}

void transformer_embedding_destroy(transformer_embedding_t *transformer_embedding)
{
    embedding_destroy(transformer_embedding->token_embedding);
    embedding_destroy(transformer_embedding->position_embedding);
    free(transformer_embedding);
}

nw_error_t *causal_multihead_self_attention_create(causal_multihead_self_attention_t **causal_multihead_self_attention, tensor_t *input_weights, tensor_t *input_bias, tensor_t *output_weights,
                                                   tensor_t *output_bias, int64_t number_of_heads, int64_t embedding_size, void *dropout_probability, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(causal_multihead_self_attention, "causal_multihead_self_attention");
    CHECK_NULL_ARGUMENT(input_weights, "input_weights");
    CHECK_NULL_ARGUMENT(output_weights, "output_weights");

    *causal_multihead_self_attention = (causal_multihead_self_attention_t *) malloc(sizeof(causal_multihead_self_attention_t));
    if (!*causal_multihead_self_attention)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(causal_multihead_self_attention_t)), NULL);
    }

    (*causal_multihead_self_attention)->dropout_probability = (void *) malloc(datatype_size(datatype));
    if (!(*causal_multihead_self_attention)->dropout_probability)
    {
        free(*causal_multihead_self_attention);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size(datatype)), NULL);
    }

    memcpy((*causal_multihead_self_attention)->dropout_probability, dropout_probability, datatype_size(datatype));
    (*causal_multihead_self_attention)->inference = false;
    (*causal_multihead_self_attention)->number_of_heads = number_of_heads;
    (*causal_multihead_self_attention)->embedding_size = embedding_size;
    (*causal_multihead_self_attention)->input_weights = input_weights;
    (*causal_multihead_self_attention)->input_bias = input_bias;
    (*causal_multihead_self_attention)->output_weights = output_weights;
    (*causal_multihead_self_attention)->output_bias = output_bias;
    (*causal_multihead_self_attention)->datatype = datatype;

    return NULL;
}

void causal_multihead_self_attention_destroy(causal_multihead_self_attention_t *causal_multihead_self_attention)
{
    if (causal_multihead_self_attention)
    {
        tensor_destroy(causal_multihead_self_attention->input_weights);
        tensor_destroy(causal_multihead_self_attention->input_bias);
        tensor_destroy(causal_multihead_self_attention->output_weights);
        tensor_destroy(causal_multihead_self_attention->output_bias);
        free(causal_multihead_self_attention->dropout_probability);
        free(causal_multihead_self_attention);
    }
}

nw_error_t *linear_layer_create(layer_t **layer, int64_t in_features, int64_t out_features, runtime_t runtime, datatype_t datatype,
                                parameter_init_t *weight_init, parameter_init_t *bias_init)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(weight_init, "weight_init");

    nw_error_t *error = NULL;
    tensor_t *weights = NULL;
    tensor_t *bias = NULL;
    linear_t *linear = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = LINEAR;
    int64_t *weight_shape = (int64_t[]) {in_features, out_features};
    int64_t *bias_shape = (int64_t[]) {out_features};
    int64_t weight_rank = 2;
    int64_t bias_rank = 1;

    error = initialize(&weights, weight_init, weight_shape, weight_rank, runtime, datatype, true);
    if (error)
    {
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
    }

    if (bias_init) 
    {
        error = initialize(&bias, bias_init, bias_shape, bias_rank, runtime, datatype, true);
        if (error)
        {
            tensor_destroy(weights);
            return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
        }
    }

    error = linear_create(&linear, weights, bias);
    if (error)
    {
        tensor_destroy(weights);
        tensor_destroy(bias);
        return ERROR(ERROR_CREATE, string_create("failed to create linear."), error);
    }

    error = transform_create(&transform, transform_type, (void *) linear);
    if (error)
    {
        linear_destroy(linear);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *linear_layer_create_from_parameters(layer_t **layer, tensor_t *weights, tensor_t *bias)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    linear_t *linear = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = LINEAR;

    error = linear_create(&linear, weights, bias);
    if (error)
    {
        tensor_destroy(weights);
        tensor_destroy(bias);
        return ERROR(ERROR_CREATE, string_create("failed to create linear."), error);
    }

    error = transform_create(&transform, transform_type, (void *) linear);
    if (error)
    {
        linear_destroy(linear);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *convolution_transpose_2d_layer_create_from_parameters(layer_t **layer, int64_t padding, int64_t stride, tensor_t *kernel, tensor_t *bias)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    convolution_2d_t *convolution_2d = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = CONVOLUTION_TRANSPOSE_2D;

    error = convolution_2d_create(&convolution_2d, padding, stride, kernel, bias);
    if (error)
    {
        tensor_destroy(kernel);
        tensor_destroy(bias);
        return ERROR(ERROR_CREATE, string_create("failed to create linear."), error);
    }

    error = transform_create(&transform, transform_type, (void *) convolution_2d);
    if (error)
    {
        convolution_2d_destroy(convolution_2d);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *convolution_transpose_2d_layer_create(layer_t **layer, int64_t kernel_size, int64_t padding, int64_t stride,
                                                  int64_t in_channels, int64_t out_channels, runtime_t runtime, datatype_t datatype,
                                                  parameter_init_t *kernel_init, parameter_init_t *bias_init)
{
    nw_error_t *error = NULL;

    error = convolution_2d_layer_create(layer, kernel_size, padding, stride, out_channels, in_channels, runtime, datatype, kernel_init, bias_init);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create convolution_2d layer."), error);
    }

    (*layer)->transform_type = CONVOLUTION_TRANSPOSE_2D;

    return error;
}

nw_error_t *convolution_2d_layer_create_from_parameters(layer_t **layer, int64_t padding, int64_t stride, tensor_t *kernel, tensor_t *bias)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    convolution_2d_t *convolution_2d = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = CONVOLUTION_2D;

    error = convolution_2d_create(&convolution_2d, padding, stride, kernel, bias);
    if (error)
    {
        tensor_destroy(kernel);
        tensor_destroy(bias);
        return ERROR(ERROR_CREATE, string_create("failed to create linear."), error);
    }

    error = transform_create(&transform, transform_type, (void *) convolution_2d);
    if (error)
    {
        convolution_2d_destroy(convolution_2d);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *convolution_2d_layer_create(layer_t **layer, int64_t kernel_size, int64_t padding, int64_t stride,
                                        int64_t in_channels, int64_t out_channels, runtime_t runtime, datatype_t datatype,
                                        parameter_init_t *kernel_init, parameter_init_t *bias_init)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(kernel_init, "kernel_init");

    nw_error_t *error = NULL;
    tensor_t *kernel = NULL;
    tensor_t *bias = NULL;
    convolution_2d_t *convolution_2d = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = CONVOLUTION_2D;
    int64_t *kernel_shape = (int64_t[]) {out_channels, in_channels, kernel_size, kernel_size};
    int64_t *bias_shape = (int64_t[]) {out_channels};
    int64_t weight_rank = 4;
    int64_t bias_rank = 1;

    error = initialize(&kernel, kernel_init, kernel_shape, weight_rank, runtime, datatype, true);
    if (error)
    {
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize kernel."), error);
    }
    
    if (bias_init)
    {
        error = initialize(&bias, bias_init, bias_shape, bias_rank, runtime, datatype, true);
        if (error)
        {
            tensor_destroy(kernel);
            return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
        }
    }

    error = convolution_2d_create(&convolution_2d, padding, stride, kernel, bias);
    if (error)
    {
        tensor_destroy(kernel);
        tensor_destroy(bias);
        return ERROR(ERROR_CREATE, string_create("failed to create convolution_2d."), error);
    }

    error = transform_create(&transform, transform_type, (void *) convolution_2d);
    if (error)
    {
        convolution_2d_destroy(convolution_2d);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *max_pooling_2d_layer_create(layer_t **layer, int64_t kernel_size, int64_t padding, int64_t stride)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    pooling_2d_t *pooling_2d = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = MAX_POOLING_2D;

    error = pooling_2d_create(&pooling_2d, padding, stride, kernel_size);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create pooling_2d layer."), error);
    }

    error = transform_create(&transform, transform_type, (void *) pooling_2d);
    if (error)
    {
        pooling_2d_destroy(pooling_2d);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *average_pooling_2d_layer_create(layer_t **layer, int64_t kernel_size, int64_t padding, int64_t stride)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    pooling_2d_t *pooling_2d = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = AVERAGE_POOLING_2D;

    error = pooling_2d_create(&pooling_2d, padding, stride, kernel_size);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create pooling_2d layer."), error);
    }

    error = transform_create(&transform, transform_type, (void *) pooling_2d);
    if (error)
    {
        pooling_2d_destroy(pooling_2d);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *dropout_layer_create(layer_t **layer, void *probability, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(probability, "probability");

    nw_error_t *error = NULL;
    dropout_t *dropout = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = DROPOUT;

    error = dropout_create(&dropout, probability, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create dropout layer."), error);
    }

    error = transform_create(&transform, transform_type, (void *) dropout);
    if (error)
    {
        dropout_destroy(dropout);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *batch_normalization_2d_layer_create(layer_t **layer, int64_t number_of_features,
                                                void *momentum, void *epsilon, bool_t track_running_stats,
                                                bool_t affine, datatype_t datatype, runtime_t runtime)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(epsilon, "epsilon");

    nw_error_t *error = NULL;
    batch_normalization_2d_t *batch_normalization_2d = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = BATCH_NORMALIZATION_2D;

    error = batch_normalization_2d_create(&batch_normalization_2d, number_of_features, momentum, epsilon, track_running_stats, affine, datatype, runtime);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create batch normalization layer."), error);
    }

    error = transform_create(&transform, transform_type, (void *) batch_normalization_2d);
    if (error)
    {
        batch_normalization_2d_destroy(batch_normalization_2d);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *layer_normalization_layer_create(layer_t **layer, const int64_t *normalized_shape, int64_t length,
                                             void *epsilon, bool_t weights, bool_t bias, datatype_t datatype, runtime_t runtime)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(normalized_shape, "normalized_shape");
    CHECK_NULL_ARGUMENT(epsilon, "epsilon");

    nw_error_t *error = NULL;
    layer_normalization_t *layer_normalization = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = LAYER_NORMALIZATION;

    error = layer_normalization_create(&layer_normalization, normalized_shape, length, epsilon, weights, bias, datatype, runtime);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create batch normalization layer."), error);
    }

    error = transform_create(&transform, transform_type, (void *) layer_normalization);
    if (error)
    {
        layer_normalization_destroy(layer_normalization);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *reshape_layer_create(layer_t **layer, int64_t *shape, int64_t length)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    reshape_t *reshape = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = RESHAPE;

    error = reshape_create(&reshape, shape, length);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create reshape layer."), error);
    }

    error = transform_create(&transform, transform_type, (void *) reshape);
    if (error)
    {
        reshape_destroy(reshape);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

static nw_error_t *vocabulary_count_create(tensor_t **vocabulary_count, int64_t vocabulary_size, datatype_t datatype, runtime_t runtime)
{
    nw_error_t *error = NULL;
    void *start = NULL;
    void *stop = NULL;
    void *step = NULL;
    size_t size = datatype_size(datatype);

    start = (void *) malloc(size);
    if (!start)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    stop = (void *) malloc(size);
    if (!stop)
    {
        free(start);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    step = (void *) malloc(size);
    if (!step)
    {
        free(start);
        free(stop);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) start = (float32_t) 0.0;
        *(float32_t *) stop = (float32_t) vocabulary_size;
        *(float32_t *) step = (float32_t) 1.0;
        break;
    case FLOAT64:
        *(float64_t *) start = (float64_t) 0.0;
        *(float64_t *) stop = (float64_t) vocabulary_size;
        *(float64_t *) step = (float64_t) 1.0;
        break;
    default:
        free(start);
        free(stop);
        free(step);
        return ERROR(ERROR_DATATYPE, string_create("unknown datatype."), NULL);
    }

    error = tensor_arange(vocabulary_count, start, stop, step, runtime, datatype, false, true);
    if (error)
    {
        free(start);
        free(stop);
        free(step);
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    free(start);
    free(stop);
    free(step);

    return error;
}

static nw_error_t *embedding_initialize(embedding_t **embedding, int64_t vocabulary_size, int64_t embedding_size, datatype_t datatype, runtime_t runtime, parameter_init_t *embedding_init)
{
    CHECK_NULL_ARGUMENT(embedding, "embedding");
    CHECK_NULL_ARGUMENT(embedding_init, "embedding_init");

    nw_error_t *error = NULL;
    tensor_t *weights = NULL;
    int64_t *embedding_shape = (int64_t[]) {vocabulary_size, embedding_size};
    int64_t embedding_rank = 2;
    tensor_t *vocabulary_count = NULL;

    error = vocabulary_count_create(&vocabulary_count, vocabulary_size, datatype, runtime);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = initialize(&weights, embedding_init, embedding_shape, embedding_rank, runtime, datatype, true);
    if (error)
    {
        tensor_destroy(vocabulary_count);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize embedding."), error);
    }

    error = embedding_create(embedding, vocabulary_size, embedding_size, vocabulary_count, weights);
    if (error)
    {
        tensor_destroy(vocabulary_count);
        tensor_destroy(weights);
        return ERROR(ERROR_CREATE, string_create("failed to create linear."), error);
    }
    
    return error;
}

nw_error_t *embedding_layer_create(layer_t **layer, int64_t vocabulary_size, int64_t embedding_size, datatype_t datatype, runtime_t runtime, parameter_init_t *embedding_init)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    embedding_t *embedding = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = EMBEDDING;

    error = embedding_initialize(&embedding, vocabulary_size, embedding_size, datatype, runtime, embedding_init);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create embedding."), error);
    }

    error = transform_create(&transform, transform_type, (void *) embedding);
    if (error)
    {
        embedding_destroy(embedding);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

static nw_error_t *embedding_create_from_parameters(embedding_t **embedding, tensor_t *weights)
{
    CHECK_NULL_ARGUMENT(embedding, "embedding");
    CHECK_NULL_ARGUMENT(weights, "weights");

    nw_error_t *error = NULL;
    tensor_t *vocabulary_count = NULL;
    int64_t vocabulary_size = weights->buffer->view->shape[0];
    int64_t embedding_size = weights->buffer->view->shape[1];
    datatype_t datatype = weights->buffer->storage->datatype;
    runtime_t runtime = weights->buffer->storage->runtime;

    error = vocabulary_count_create(&vocabulary_count, vocabulary_size, datatype, runtime);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
    }

    error = embedding_create(embedding, vocabulary_size, embedding_size, vocabulary_count, weights);
    if (error)
    {
        tensor_destroy(vocabulary_count);
        return ERROR(ERROR_CREATE, string_create("failed to create linear."), error);
    }

    return error;
}

nw_error_t *embedding_layer_create_from_parameters(layer_t **layer, tensor_t *weights)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(weights, "weights");

    nw_error_t *error = NULL;
    embedding_t *embedding = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = EMBEDDING;

    error = embedding_create_from_parameters(&embedding,  weights);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create linear."), error);
    }

    error = transform_create(&transform, transform_type, (void *) embedding);
    if (error)
    {
        embedding_destroy(embedding);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *transformer_embedding_layer_create(layer_t **layer, int64_t vocabulary_size, int64_t embedding_size, int64_t block_size, datatype_t datatype, runtime_t runtime,
                                               parameter_init_t *token_embedding_init, parameter_init_t *position_embedding_init)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(token_embedding_init, "token_embedding_init");
    CHECK_NULL_ARGUMENT(position_embedding_init, "position_embedding_init");

    nw_error_t *error = NULL;
    embedding_t *token_embedding = NULL;
    embedding_t *position_embedding = NULL;
    transform_t *transform = NULL;
    transformer_embedding_t *transformer_embedding = NULL;
    transform_type_t transform_type = TRANSFORMER_EMBEDDING;

    error = embedding_initialize(&token_embedding, vocabulary_size, embedding_size, datatype, runtime, token_embedding_init);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create embedding."), error);
    }

    error = embedding_initialize(&position_embedding, block_size, embedding_size, datatype, runtime, position_embedding_init);
    if (error)
    {
        embedding_destroy(token_embedding);
        return ERROR(ERROR_CREATE, string_create("failed to create embedding."), error);
    }

    error = transformer_embedding_create(&transformer_embedding, token_embedding, position_embedding);
    if (error)
    {
        embedding_destroy(token_embedding);
        embedding_destroy(position_embedding);
        return ERROR(ERROR_CREATE, string_create("failed to create transformer embedding."), error);
    }

    error = transform_create(&transform, transform_type, (void *) transformer_embedding);
    if (error)
    {
        transformer_embedding_destroy(transformer_embedding);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *transformer_embedding_layer_create_from_parameters(layer_t **layer, tensor_t *token_weights, tensor_t *position_weights)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(token_weights, "token_weights");
    CHECK_NULL_ARGUMENT(position_weights, "position_weights");

    nw_error_t *error = NULL;
    embedding_t *token_embedding = NULL;
    embedding_t *position_embedding = NULL;
    transform_t *transform = NULL;
    transformer_embedding_t *transformer_embedding = NULL;
    transform_type_t transform_type = TRANSFORMER_EMBEDDING;

    error = embedding_create_from_parameters(&token_embedding, token_weights);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create embedding."), error);
    }

    error = embedding_create_from_parameters(&position_embedding, position_weights);
    if (error)
    {
        embedding_destroy(token_embedding);
        return ERROR(ERROR_CREATE, string_create("failed to create embedding."), error);
    }

    error = transformer_embedding_create(&transformer_embedding, token_embedding, position_embedding);
    if (error)
    {
        embedding_destroy(token_embedding);
        embedding_destroy(position_embedding);
        return ERROR(ERROR_CREATE, string_create("failed to create transformer embedding."), error);
    }

    error = transform_create(&transform, transform_type, (void *) transformer_embedding);
    if (error)
    {
        transformer_embedding_destroy(transformer_embedding);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *causal_multihead_self_attention_layer_create(layer_t **layer, int64_t number_of_heads, int64_t embedding_size, void *dropout_probability, datatype_t datatype, runtime_t runtime,
                                                         parameter_init_t *input_weight_init, parameter_init_t *input_bias_init, parameter_init_t *output_weight_init, parameter_init_t *output_bias_init)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(dropout_probability, "dropout_probability");
    CHECK_NULL_ARGUMENT(input_weight_init, "input_weight_init");
    CHECK_NULL_ARGUMENT(output_weight_init, "output_weight_init");

    nw_error_t *error = NULL;
    tensor_t *input_weights = NULL;
    tensor_t *input_bias = NULL;
    tensor_t *output_weights = NULL;
    tensor_t *output_bias = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = CAUSAL_MULTIHEAD_SELF_ATTENTION;
    causal_multihead_self_attention_t *causal_multihead_self_attention = NULL;

    int64_t *input_weight_shape = (int64_t[]) {embedding_size, 3 * embedding_size};
    int64_t *input_bias_shape = (int64_t[]) {3 * embedding_size};
    int64_t *output_weight_shape = (int64_t[]) {embedding_size, embedding_size};
    int64_t *output_bias_shape = (int64_t[]) {embedding_size};
    int64_t weight_rank = 2;
    int64_t bias_rank = 1;

    error = initialize(&input_weights, input_weight_init, input_weight_shape, weight_rank, runtime, datatype, true);
    if (error)
    {
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
    }

    if (input_bias_init) 
    {
        error = initialize(&input_bias, input_bias_init, input_bias_shape, bias_rank, runtime, datatype, true);
        if (error)
        {
            tensor_destroy(input_weights);
            return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
        }
    }

    error = initialize(&output_weights, output_weight_init, output_weight_shape, weight_rank, runtime, datatype, true);
    if (error)
    {
        tensor_destroy(input_weights);
        tensor_destroy(input_bias);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
    }

    if (output_bias_init) 
    {
        error = initialize(&output_bias, output_bias_init, output_bias_shape, bias_rank, runtime, datatype, true);
        if (error)
        {
            tensor_destroy(input_weights);
            tensor_destroy(input_bias);
            tensor_destroy(output_weights);
            return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
        }
    }

    error = causal_multihead_self_attention_create(&causal_multihead_self_attention, input_weights, input_bias, output_weights, 
                                                   output_bias, number_of_heads, embedding_size, dropout_probability, datatype);
    if (error)
    {
        tensor_destroy(input_weights);
        tensor_destroy(input_bias);
        tensor_destroy(output_weights);
        tensor_destroy(output_bias);
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
    }

    error = transform_create(&transform, transform_type, (void *) causal_multihead_self_attention);
    if (error)
    {
        causal_multihead_self_attention_destroy(causal_multihead_self_attention);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *causal_multihead_self_attention_layer_create_from_parameters(layer_t **layer, int64_t number_of_heads, int64_t embedding_size, void *dropout_probability, datatype_t datatype,
                                                                         tensor_t *input_weights, tensor_t *input_bias, tensor_t *output_weights, tensor_t *output_bias)
{
    nw_error_t *error = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = CAUSAL_MULTIHEAD_SELF_ATTENTION;
    causal_multihead_self_attention_t *causal_multihead_self_attention = NULL;

    error = causal_multihead_self_attention_create(&causal_multihead_self_attention, input_weights, input_bias, output_weights, 
                                                   output_bias, number_of_heads, embedding_size, dropout_probability, datatype);
    if (error)
    {
        return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
    }

    error = transform_create(&transform, transform_type, (void *) causal_multihead_self_attention);
    if (error)
    {
        causal_multihead_self_attention_destroy(causal_multihead_self_attention);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *residual_block_layer_create(layer_t **layer, block_t *block)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = RESIDUAL_BLOCK;

    error = transform_create(&transform, transform_type, (void *) block);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *block_layer_create(layer_t **layer, block_t *block)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = BLOCK;

    error = transform_create(&transform, transform_type, (void *) block);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *rectified_linear_activation_layer_create(layer_t **layer)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    activation_t *activation = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = ACTIVATION;

    error = rectified_linear_activation_create(&activation);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    error = transform_create(&transform, transform_type, (void *) activation);
    if (error)
    {
        activation_destroy(activation);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *sigmoid_activation_layer_create(layer_t **layer)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    activation_t *activation = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = ACTIVATION;

    error = sigmoid_activation_create(&activation);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    error = transform_create(&transform, transform_type, (void *) activation);
    if (error)
    {
        activation_destroy(activation);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *softmax_activation_layer_create(layer_t **layer, int64_t axis)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    activation_t *activation = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = ACTIVATION;

    error = softmax_activation_create(&activation, axis);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    error = transform_create(&transform, transform_type, (void *) activation);
    if (error)
    {
        activation_destroy(activation);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *logsoftmax_activation_layer_create(layer_t **layer, int64_t axis)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    activation_t *activation = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = ACTIVATION;

    error = logsoftmax_activation_create(&activation, axis);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    error = transform_create(&transform, transform_type, (void *) activation);
    if (error)
    {
        activation_destroy(activation);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *leaky_rectified_linear_activation_layer_create(layer_t **layer, void *c, datatype_t datatype)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(c, "c");

    nw_error_t *error = NULL;
    activation_t *activation = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = ACTIVATION;

    error = leaky_rectified_linear_activation_create(&activation, c, datatype);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    error = transform_create(&transform, transform_type, (void *) activation);
    if (error)
    {
        activation_destroy(activation);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *tanh_activation_layer_create(layer_t **layer)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    activation_t *activation = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = ACTIVATION;

    error = tanh_activation_create(&activation);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    error = transform_create(&transform, transform_type, (void *) activation);
    if (error)
    {
        activation_destroy(activation);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

nw_error_t *gelu_activation_layer_create(layer_t **layer)
{
    CHECK_NULL_ARGUMENT(layer, "layer");

    nw_error_t *error = NULL;
    activation_t *activation = NULL;
    transform_t *transform = NULL;
    transform_type_t transform_type = ACTIVATION;

    error = gelu_activation_create(&activation);
    if (error)
    {
        return ERROR(ERROR_CREATE, string_create("failed to create activation."), error);
    }

    error = transform_create(&transform, transform_type, (void *) activation);
    if (error)
    {
        activation_destroy(activation);
        return ERROR(ERROR_CREATE, string_create("failed to create transform."), error);
    }

    error = layer_create(layer, transform, transform_type);
    if (error)
    {
        transform_destroy(transform, transform_type);
        return ERROR(ERROR_CREATE, string_create("failed to create layer."), error);
    }

    return error;
}

//insert create
nw_error_t *rnn_cell_create(rnn_cell_t **rnn, activation_function_type_t activation, tensor_t *weight_ih, tensor_t *bias_ih, tensor_t *weight_hh, tensor_t *bias_hh)
{
    CHECK_NULL_ARGUMENT(rnn, "rnn");
    CHECK_NULL_ARGUMENT(weight_ih, "weight_ih");
    CHECK_NULL_ARGUMENT(weight_hh, "weight_hh");
    CHECK_NULL_ARGUMENT(activation, "activation");

    *rnn = (rnn_cell_t *) malloc(sizeof(rnn_cell_t));
    if (!*rnn)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(rnn_cell_t)), NULL);
    }

    (*rnn)->weight_ih = weight_ih;
    (*rnn)->bias_ih = bias_ih;
    (*rnn)->weight_hh = weight_hh;
    (*rnn)->weight_ih = weight_ih;
    (*rnn)->bias_hh = bias_hh;
    (*rnn)->activation = activation;

    return NULL;
}

void rnn_cell_destroy(rnn_cell_t *cell)
{
    if (cell)
    {
        tensor_destroy(cell->weight_ih);
        tensor_destroy(cell->bias_ih);
        tensor_destroy(cell->weight_hh);
        tensor_destroy(cell->bias_hh);
    free(cell);
    }
}

nw_error_t *gru_cell_create(gru_cell_t **gru, tensor_t *weight_ir, tensor_t *weight_iz, tensor_t *weight_in, tensor_t *weight_hr, tensor_t *weight_hz, tensor_t *weight_hn, 
                            tensor_t *bias_ir, tensor_t *bias_iz, tensor_t *bias_in, tensor_t *bias_hr, tensor_t *bias_hz, tensor_t *bias_hn)
{
    CHECK_NULL_ARGUMENT(gru, "gru");
    CHECK_NULL_ARGUMENT(weight_ir, "weight_ir");
    CHECK_NULL_ARGUMENT(weight_iz, "weight_iz");
    CHECK_NULL_ARGUMENT(weight_in, "weight_in");
    CHECK_NULL_ARGUMENT(weight_hr, "weight_hr");
    CHECK_NULL_ARGUMENT(weight_hz, "weight_hz");
    CHECK_NULL_ARGUMENT(weight_hn, "weight_hn");

    *gru = (gru_cell_t *) malloc(sizeof(gru_cell_t));
    if (!*gru)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(gru_cell_t)), NULL);
    }

    (*gru)->weight_ir = weight_ir;
    (*gru)->weight_iz = weight_iz;
    (*gru)->weight_in = weight_in;
    (*gru)->weight_hr = weight_hr;
    (*gru)->weight_hz = weight_hz;
    (*gru)->weight_hn = weight_hn;
    (*gru)->bias_ir = bias_ir;
    (*gru)->bias_iz = bias_iz;
    (*gru)->bias_in = bias_in;
    (*gru)->bias_hr = bias_hr;
    (*gru)->bias_hz = bias_hz;
    (*gru)->bias_hn = bias_hn;

    return NULL;             
}

void gru_cell_destroy(gru_cell_t *cell)
{
    if (cell) 
    {
        tensor_destroy(cell->weight_ir);
        tensor_destroy(cell->weight_iz);
        tensor_destroy(cell->weight_in);
        tensor_destroy(cell->weight_hr);
        tensor_destroy(cell->weight_hz);
        tensor_destroy(cell->weight_hn);
        tensor_destroy(cell->bias_ir);
        tensor_destroy(cell->bias_iz);
        tensor_destroy(cell->bias_in);
        tensor_destroy(cell->bias_hr);
        tensor_destroy(cell->bias_hz);
        tensor_destroy(cell->bias_hn);

        free(cell);
    }
}

nw_error_t *lstm_cell_create(lstm_cell_t **lstm, tensor_t *weight_xi, tensor_t *weight_hi, tensor_t *bias_xi, tensor_t *bias_hi, tensor_t *weight_xf, tensor_t *weight_hf, tensor_t *bias_xf, tensor_t *bias_hf, 
                            tensor_t *weight_xc, tensor_t *weight_hc, tensor_t *bias_xc, tensor_t *bias_hc, tensor_t *weight_xo, tensor_t *weight_ho, tensor_t *bias_xo, tensor_t *bias_ho, tensor_t *cell_state)
{
    CHECK_NULL_ARGUMENT(lstm, "lstm");
    CHECK_NULL_ARGUMENT(weight_xi, "weight_xi");
    CHECK_NULL_ARGUMENT(weight_hi, "weight_hi");
    CHECK_NULL_ARGUMENT(weight_xf, "weight_xf");
    CHECK_NULL_ARGUMENT(weight_hf, "weight_hf");
    CHECK_NULL_ARGUMENT(weight_xc, "weight_xc");
    CHECK_NULL_ARGUMENT(weight_hc, "weight_hc");
    CHECK_NULL_ARGUMENT(weight_xo, "weight_xo");
    CHECK_NULL_ARGUMENT(weight_ho, "weight_ho");
    CHECK_NULL_ARGUMENT(cell_state, "cell_state");

    *lstm = (lstm_cell_t *) malloc(sizeof(lstm_cell_t));
    if (!*lstm)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(lstm_cell_t)), NULL);
    }

    (*lstm)->weight_xi = weight_xi;
    (*lstm)->weight_hi = weight_hi;
    (*lstm)->bias_xi = bias_xi;
    (*lstm)->bias_hi = bias_hi;
    (*lstm)->weight_xf = weight_xf;
    (*lstm)->weight_hf = weight_hf;
    (*lstm)->bias_xf = bias_xf;
    (*lstm)->bias_hf = bias_hf;
    (*lstm)->weight_xc = weight_xc;
    (*lstm)->weight_hc = weight_hc;
    (*lstm)->bias_xc = bias_xc;
    (*lstm)->bias_hc = bias_hc;
    (*lstm)->weight_xo = weight_xo;
    (*lstm)->weight_ho = weight_ho;
    (*lstm)->bias_xo = bias_xo;
    (*lstm)->bias_ho = bias_ho;
    (*lstm)->cell_state = cell_state;

    return NULL; 
}

void lstm_cell_destroy(lstm_cell_t *cell)
{
    if (cell) 
    {
        tensor_destroy(cell->weight_xi);
        tensor_destroy(cell->weight_hi);
        tensor_destroy(cell->bias_xi);
        tensor_destroy(cell->bias_hi);

        tensor_destroy(cell->weight_xf);
        tensor_destroy(cell->weight_hf);
        tensor_destroy(cell->bias_xf);
        tensor_destroy(cell->bias_hf);

        tensor_destroy(cell->weight_xc);
        tensor_destroy(cell->weight_hc);
        tensor_destroy(cell->bias_xc);
        tensor_destroy(cell->bias_hc);

        tensor_destroy(cell->weight_xo);
        tensor_destroy(cell->weight_ho);
        tensor_destroy(cell->bias_xo);
        tensor_destroy(cell->bias_ho);

        tensor_destroy(cell->cell_state);

        free(cell);
    }
}

nw_error_t *rnn_layer_create(rnn_layer_t **layer, recurrent_type_t type, int64_t batch_size, int64_t hidden_size, int64_t input_size, runtime_t runtime, datatype_t datatype, bool_t bidirectional,
                             parameter_init_t *weight_init, parameter_init_t *bias_init, activation_function_type_t activation)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(weight_init, "weight_init");

   *layer = (rnn_layer_t *) malloc(sizeof(rnn_layer_t));
   if (!*layer)
   {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytees.", sizeof(rnn_layer_t)), NULL);
   }

   nw_error_t *error = NULL;

   (*layer)->type = type;
   (*layer)->hidden_size = hidden_size;
   (*layer)->input_size = input_size;
   (*layer)->bidirectional = bidirectional;

   map_t *map = NULL;
   error = map_create(&map);
   (*layer)->cells = map;

   string_t index;
   for (int64_t i = 0; i < hidden_size; ++i)
   {
        index = string_create("%lu", i);

        switch(type)
        {
            case RNN:
                rnn_cell_t *rnn = NULL;
                tensor_t *weight_ih, *bias_ih, *weight_hh, *bias_hh;
                int64_t *weight_ih_shape, *weight_hh_shape, weight_ih_rank, weight_hh_rank, *bias_ih_shape, *bias_hh_shape, bias_ih_rank, bias_hh_rank;

                weight_ih_rank = 2;
                weight_ih_shape = (int64_t[]) {hidden_size, input_size};

                weight_hh_rank = 2;
                weight_hh_shape = (int64_t[]) {hidden_size, hidden_size};

                bias_ih_rank = 1;
                bias_ih_shape = (int64_t[]) {hidden_size};

                bias_hh_rank = 1;
                bias_hh_shape = (int64_t[]) {hidden_size};

                error = initialize(&weight_ih, weight_init, weight_ih_shape, weight_ih_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }

                error = initialize(&weight_hh, weight_init, weight_hh_shape, weight_hh_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }

                error = initialize(&bias_ih, weight_init, bias_ih_shape, bias_ih_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                }

                error = initialize(&bias_hh, weight_init, bias_hh_shape, bias_hh_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                }

                error = rnn_cell_create(&rnn, activation, weight_ih, bias_ih, weight_hh, bias_hh);

                error = map_set((*layer)->cells, index, rnn);
                break;
            case GRU:
                gru_cell_t *gru = NULL;
                tensor_t *weight_ir, *weight_iz, *weight_in, *weight_hr, *weight_hz, *weight_hn;
                tensor_t *bias_ir, *bias_iz, *bias_in, *bias_hr, *bias_hz, *bias_hn;
                int64_t *weight_ir_shape, *weight_iz_shape, *weight_in_shape, *weight_hr_shape, *weight_hz_shape, *weight_hn_shape;
                int64_t *bias_ir_shape, *bias_iz_shape, *bias_in_shape, *bias_hr_shape, *bias_hz_shape, *bias_hn_shape;
                int64_t weight_ir_rank, weight_iz_rank, weight_in_rank, weight_hr_rank, weight_hz_rank, weight_hn_rank;
                int64_t bias_ir_rank, bias_iz_rank, bias_in_rank, bias_hr_rank, bias_hz_rank, bias_hn_rank;

                weight_ir_rank = 2;
                weight_iz_rank = 2;
                weight_in_rank = 2;
                weight_hr_rank = 2;
                weight_hz_rank = 2;
                weight_hn_rank = 2;
                bias_ir_rank = 1;
                bias_iz_rank = 1;
                bias_in_rank = 1;
                bias_hr_rank = 1;
                bias_hz_rank = 1;
                bias_hn_rank = 1;

                weight_ir_shape = (int64_t[]) {hidden_size, input_size};
                weight_iz_shape = (int64_t[]) {hidden_size, input_size};
                weight_in_shape = (int64_t[]) {hidden_size, input_size};

                weight_hr_shape = (int64_t[]) {hidden_size, hidden_size};
                weight_hz_shape = (int64_t[]) {hidden_size, hidden_size};
                weight_hn_shape = (int64_t[]) {hidden_size, hidden_size};

                bias_ir_shape = (int64_t[]) {hidden_size};
                bias_iz_shape = (int64_t[]) {hidden_size};
                bias_in_shape = (int64_t[]) {hidden_size};
                bias_hr_shape = (int64_t[]) {hidden_size};
                bias_hz_shape = (int64_t[]) {hidden_size};
                bias_hn_shape = (int64_t[]) {hidden_size};


                error = initialize(&weight_ir, weight_init, weight_ir_shape, weight_ir_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }

                error = initialize(&weight_iz, weight_init, weight_iz_shape, weight_iz_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }

                error = initialize(&weight_in, weight_init, weight_in_shape, weight_in_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }

                error = initialize(&weight_hr, weight_init, weight_hr_shape, weight_hr_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }

                error = initialize(&weight_hz, weight_init, weight_hz_shape, weight_hz_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }

                error = initialize(&weight_hn, weight_init, weight_hn_shape, weight_hn_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }

                
                error = initialize(&bias_ir, weight_init, bias_ir_shape, bias_ir_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                }

                error = initialize(&bias_iz, weight_init, bias_iz_shape, bias_iz_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                }

                error = initialize(&bias_in, bias_init, bias_in_shape, bias_in_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                }

                error = initialize(&bias_hr, bias_init, bias_hr_shape, bias_hr_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                }

                error = initialize(&bias_hz, bias_init, bias_hz_shape, bias_hz_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                }

                error = initialize(&bias_hn, bias_init, bias_hn_shape, bias_hn_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                }

                error = gru_cell_create(&gru, weight_ir, weight_iz, weight_in, weight_hr, weight_hz, weight_hn, 
                                        bias_ir, bias_iz, bias_in, bias_hr, bias_hz, bias_hn);

                error = map_set((*layer)->cells, index, gru);
                break;
            case LSTM:
                lstm_cell_t *lstm = NULL;
                tensor_t *weight_xi, *weight_hi, *bias_xi, *bias_hi;
                tensor_t *weight_xf, *weight_hf, *bias_xf, *bias_hf;
                tensor_t *weight_xc, *weight_hc, *bias_xc, *bias_hc;
                tensor_t *weight_xo, *weight_ho, *bias_xo, *bias_ho;
                tensor_t *cell_state;
                int64_t *weight_xi_shape, *weight_hi_shape, *bias_xi_shape, *bias_hi_shape;
                int64_t *weight_xf_shape, *weight_hf_shape, *bias_xf_shape, *bias_hf_shape;
                int64_t *weight_xc_shape, *weight_hc_shape, *bias_xc_shape, *bias_hc_shape;
                int64_t *weight_xo_shape, *weight_ho_shape, *bias_xo_shape, *bias_ho_shape;
                int64_t *cell_state_shape;
                int64_t weight_xi_rank, weight_hi_rank, bias_xi_rank, bias_hi_rank;
                int64_t weight_xf_rank, weight_hf_rank, bias_xf_rank, bias_hf_rank;
                int64_t weight_xc_rank, weight_hc_rank, bias_xc_rank, bias_hc_rank;
                int64_t weight_xo_rank, weight_ho_rank, bias_xo_rank, bias_ho_rank;
                int64_t cell_state_rank;

                
                weight_xi_rank = 2;
                weight_hi_rank = 2;
                bias_xi_rank = 1;
                bias_hi_rank = 1;
                weight_xf_rank = 2;
                weight_hf_rank = 2;
                bias_xf_rank = 1;
                bias_hf_rank = 1;
                weight_xc_rank = 2;
                weight_hc_rank = 2;
                bias_xc_rank = 1;
                bias_hc_rank = 1;
                weight_xo_rank = 2;
                weight_ho_rank = 2;
                bias_xo_rank = 1;
                bias_ho_rank = 1;
                cell_state_rank = 2;


                weight_xi_shape[0] = (int64_t[]) {hidden_size, input_size};
                weight_xi_shape[1] = hidden_size;

                weight_hi_shape[0] = hidden_size;
                weight_hi_shape[1] = hidden_size;

                bias_xi_shape[0] = hidden_size;

                bias_hi_shape[0] = hidden_size;

                weight_xf_shape[0] = input_size;
                weight_xf_shape[1] = hidden_size;

                weight_hf_shape[0] = hidden_size;
                weight_hf_shape[1] = hidden_size;

                bias_xf_shape[0] = hidden_size;

                bias_hf_shape[0] = hidden_size;

                weight_xc_shape[0] = input_size;
                weight_xc_shape[1] = hidden_size;

                weight_hc_shape[0] = hidden_size;
                weight_hc_shape[1] = hidden_size;

                bias_xc_shape[0] = hidden_size;

                bias_hc_shape[0] = hidden_size;

                weight_xo_shape[0] = input_size;
                weight_xo_shape[1] = hidden_size;

                weight_ho_shape[0] = hidden_size;
                weight_ho_shape[1] = hidden_size;

                bias_xo_shape[0] = hidden_size;
                bias_ho_shape[0] = hidden_size;

                cell_state_shape = (int64_t *) malloc(2 * sizeof(int64_t));
                cell_state_shape[0] = batch_size;
                cell_state_shape[1] = hidden_size;

                error = initialize(&weight_xi, weight_init, weight_xi_shape, weight_xi_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }

                error = initialize(&weight_hi, weight_init, weight_hi_shape, weight_hi_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }

                error = initialize(&bias_xi, weight_init, bias_xi_shape, bias_xi_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                }

                error = initialize(&bias_hi, weight_init, bias_hi_shape, bias_hi_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                }

                error = initialize(&weight_xf, weight_init, weight_xf_shape, weight_xf_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }

                error = initialize(&weight_hf, weight_init, weight_hf_shape, weight_hf_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }

                error = initialize(&bias_xf, weight_init, bias_xf_shape, bias_xf_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                }

                error = initialize(&bias_hf, weight_init, bias_hf_shape, bias_hf_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                }
                
                error = initialize(&weight_xc, weight_init, weight_xc_shape, weight_xc_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }

                error = initialize(&weight_hc, weight_init, weight_hc_shape, weight_hc_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }
                                
                error = initialize(&bias_xc, weight_init, bias_xc_shape, bias_xc_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                }

                error = initialize(&bias_hc, weight_init, bias_hc_shape, bias_hc_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                }
                
                error = initialize(&weight_xo, weight_init, weight_xo_shape, weight_xo_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }

                error = initialize(&weight_ho, weight_init, weight_ho_shape, weight_ho_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize weights."), error);
                }

                error = initialize(&bias_xo, weight_init, bias_xo_shape, bias_xo_rank, runtime, datatype, true);
                if (error)
                {
                    return ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                }

                error = initialize(&bias_ho, weight_init, bias_ho_shape, bias_ho_rank, runtime, datatype, true);
                if (error)
                {
                    error = ERROR(ERROR_INITIALIZATION, string_create("failed to initialize bias."), error);
                    goto cleanup;
                }

                error = tensor_create_zeroes(&cell_state, cell_state_shape, cell_state_rank, runtime, datatype, true, true); // TODO: Allow user to init this?

                error = lstm_cell_create(&lstm, weight_xi, weight_hi, bias_xi, bias_hi, 
                                        weight_xf, weight_hf, bias_xf, bias_hf, 
                                        weight_xc, weight_hc, bias_xc, bias_hc,
                                        weight_xo, weight_ho, bias_xo, bias_ho,
                                        cell_state);
                error = map_set((*layer)->cells, index, lstm);
                break;
            default:
                // error no type
                break;
        }
   }
    return error;

cleanup:
    return error;
}

void rnn_layer_destroy(rnn_layer_t *layer)
{
    if(layer)
    {
        for (uint64_t i = 0; i < layer->cells->capacity; ++i)
        {
            if (layer->cells->entries[i].data)
            {
                switch(layer->type)
                {
                    case RNN:
                        rnn_cell_destroy((rnn_cell_t *) layer->cells->entries[i].data);
                        break;
                    case GRU:
                        gru_cell_destroy((gru_cell_t *) layer->cells->entries[i].data);
                        break;
                    case LSTM:
                        lstm_cell_destroy((lstm_cell_t *) layer->cells->entries[i].data);
                        break;
                    default:
                        break; // TODO: What to do here?

                }
            }
        }
        map_destroy(layer->cells);
        free(layer);
    }
}

nw_error_t *rnn_stack_create(rnn_stack_t **stack, recurrent_type_t type, int64_t num_layers, int64_t batch_size, int64_t input_size, int64_t hidden_size, runtime_t runtime, datatype_t datatype, bool_t bidirectional,
                             parameter_init_t *weight_init, parameter_init_t *bias_init, activation_function_type_t activation, void *dropout)
{
    CHECK_NULL_ARGUMENT(stack, "stack");
    CHECK_NULL_ARGUMENT(weight_init, "weight_init");
    CHECK_NULL_ARGUMENT(bias_init, "bias_init");

    *stack = (rnn_stack_t *) malloc(sizeof(rnn_stack_t));
    if (!*stack)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytees.", sizeof(rnn_stack_t)), NULL);
    }

    nw_error_t *error = NULL;

    (*stack)->type = type;
    (*stack)->hidden_size = hidden_size;
    (*stack)->input_size = input_size;
    (*stack)->bidirectional = bidirectional;

    (*stack)->dropout = (void *) malloc(datatype_size(datatype));
    if (!(*stack)->dropout)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size(datatype)), NULL);
        goto cleanup;
    }
    memcpy((*stack)->dropout, dropout, datatype_size(datatype));

    map_t *map = NULL;
    error = map_create(&map);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create map"), NULL);
        goto cleanup;
    }
    (*stack)->layers = map;

    string_t index;
    for (int64_t i = 0; i < num_layers; ++i)
    {
        index = string_create("%lu", i);
        rnn_layer_t *layer = NULL;
        switch(type)
        {
            case RNN:
                error = rnn_layer_create(&layer, type, batch_size, hidden_size, input_size, runtime, datatype, bidirectional, weight_init, bias_init, activation);
                if (error)
                {
                    error = ERROR(ERROR_RNN, string_create("failed to create rnn layer"), error);
                    goto cleanup;
                }

                error = map_set((*stack)->layers, index, layer);
                if (error)
                {
                    error = ERROR(ERROR_SET, string_create("failed to set entry."), error);
                    goto cleanup;
                }
                break;
            case GRU:
                error = rnn_layer_create(&layer, type, batch_size, hidden_size, input_size, runtime, datatype, bidirectional, weight_init, bias_init, activation);
                if (error)
                {
                    error = ERROR(ERROR_RNN, string_create("failed to create rnn layer"), error);
                    goto cleanup;
                }

                error = map_set((*stack)->layers, index, layer);
                if (error)
                {
                    error = ERROR(ERROR_SET, string_create("failed to set entry."), error);
                    goto cleanup;
                }
                break;
            case LSTM:
                error = rnn_layer_create(&layer, type, batch_size, hidden_size, input_size, runtime, datatype, bidirectional, weight_init, bias_init, activation);
                if (error)
                {
                    error = ERROR(ERROR_RNN, string_create("failed to create rnn layer"), error);
                    goto cleanup;
                }

                error = map_set((*stack)->layers, index, layer);
                if (error)
                {
                    error = ERROR(ERROR_SET, string_create("failed to set entry."), error);
                    goto cleanup;
                }

                break;
            default:
                return ERROR(ERROR_RNN, string_create("Unknown recurrent cell type."), error);
                goto cleanup;
        }
        string_destroy(index);
        index = NULL;
    }
    return error;

cleanup:
    rnn_stack_destroy(*stack);
    return error;    
}

void rnn_stack_destroy(rnn_stack_t *stack)
{
    if(stack)
    {
        for (uint64_t i = 0; i < stack->layers->capacity; ++i)
        {
            if (stack->layers->entries[i].data)
            {
                rnn_layer_destroy((rnn_layer_t *) stack->layers->entries[i].data);
            }
        }
        map_destroy(stack->layers);
        free(stack->dropout);
        free(stack);
    }
}

nw_error_t *model_forward(model_t *model, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_MODEL("model", model);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(model, "model");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = block_forward(model->block, x, y);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed forward pass"), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *block_forward(block_t *block, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_BLOCK("block", block);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(block, "block");
    CHECK_NULL_ARGUMENT(block->layers, "block->layers");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *feature_map = NULL;

    for (int64_t i = 0; i < block->depth; ++i)
    {
        layer_t *layer = block->layers[i];
        if (!layer)
        {
            return ERROR(ERROR_NULL, string_create("layer is null."), NULL);
        }

        transform_type_t transform_type = layer->transform_type;
        transform_t *transform = layer->transform;
        if (!transform)
        {
            return ERROR(ERROR_NULL, string_create("transform is null."), NULL);
        }

        switch (transform_type)
        {
        case LINEAR:
            error = linear_forward(transform->linear, x, &feature_map);
            break;
        case CONVOLUTION_2D:
            error = convolution_2d_forward(transform->convolution_2d, x, &feature_map);
            break;
        case CONVOLUTION_TRANSPOSE_2D:
            error = convolution_transpose_2d_forward(transform->convolution_2d, x, &feature_map);
            break;
        case MAX_POOLING_2D:
            error = max_pooling_2d_forward(transform->pooling_2d, x, &feature_map);
            break;
        case AVERAGE_POOLING_2D:
            error = average_pooling_2d_forward(transform->pooling_2d, x, &feature_map);
            break;
        case DROPOUT:
            error = dropout_forward(transform->dropout, x, &feature_map);
            break;
        case BATCH_NORMALIZATION_2D:
            error = batch_normalization_2d_forward(transform->batch_normalization_2d, x, &feature_map);
            break;
        case RESHAPE:
            error = reshape_forward(transform->reshape, x, &feature_map);
            break;
        case LAYER_NORMALIZATION:
            error = layer_normalization_forward(transform->layer_normalization, x, &feature_map);
            break;
        case EMBEDDING:
            error = embedding_forward(transform->embedding, x, &feature_map);
            break;
        case TRANSFORMER_EMBEDDING:
            error = transformer_embedding_forward(transform->transformer_embedding, x, &feature_map);
            break;
        case CAUSAL_MULTIHEAD_SELF_ATTENTION:
            error = causal_multihead_self_attention_forward(transform->causal_multihead_self_attention, x, &feature_map);
            break;
        case ACTIVATION:
            error = activation_forward(transform->activation, x, &feature_map);
            break;
        case RESIDUAL_BLOCK:
            error = residual_block_forward(transform->block, x, &feature_map);
            break;
        case BLOCK:
            error = block_forward(transform->block, x, &feature_map);
            break;
        default:
            error = ERROR(ERROR_LAYER_TYPE, string_create("unknown transform type %d.", (int) transform_type), NULL);
            break;
        }

        if (error)
        {
            return ERROR(ERROR_FORWARD, string_create("failed forward pass."), error);
        }

        if (i > 0 && (!feature_map->requires_gradient || no_gradient) && x != feature_map)
        {
            tensor_destroy(x);
        }

        x = feature_map;
        feature_map = NULL;
    }

    *y = x;

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *linear_forward(linear_t *linear, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_LINEAR("linear", linear);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(linear, "linear");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_linear(x, linear->weights, linear->bias, y);
    if (error)
    {
        return ERROR(ERROR_LINEAR, string_create("failed to matrix multiply tensors."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *convolution_2d_forward(convolution_2d_t *convolution_2d, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(convolution_2d, "convolution_2d");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_convolution_2d(x, convolution_2d->kernel, convolution_2d->bias, y, convolution_2d->stride, convolution_2d->padding);
    if (error)
    {
        return ERROR(ERROR_CONVOLUTION, string_create("failed to apply convolution_2d."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *convolution_transpose_2d_forward(convolution_2d_t *convolution_2d, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(convolution_2d, "convolution_2d");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_convolution_transpose_2d(x, convolution_2d->kernel, convolution_2d->bias, y, convolution_2d->stride, convolution_2d->padding);
    if (error)
    {
        return ERROR(ERROR_CONVOLUTION, string_create("failed to apply convolution_2d transpose."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *max_pooling_2d_forward(pooling_2d_t *pooling_2d, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(pooling_2d, "pooling_2d");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_max_pool_2d(x, y, pooling_2d->kernel, pooling_2d->stride, pooling_2d->padding);
    if (error)
    {
        return ERROR(ERROR_POOLING, string_create("failed to apply pooling."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *average_pooling_2d_forward(pooling_2d_t *pooling_2d, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(pooling_2d, "pooling_2d");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_average_pool_2d(x, y, pooling_2d->kernel, pooling_2d->stride, pooling_2d->padding);
    if (error)
    {
        return ERROR(ERROR_POOLING, string_create("failed to apply pooling."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;

}

nw_error_t *dropout_forward(dropout_t *dropout, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_DROPOUT("dropout", dropout);
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(dropout, "dropout");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_dropout(x, y, dropout->probability, dropout->inference);
    if (error)
    {
        return ERROR(ERROR_DROPOUT, string_create("failed to apply dropout."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return NULL;
}

nw_error_t *batch_normalization_2d_forward(batch_normalization_2d_t *batch_normalization_2d, tensor_t *x, tensor_t **y)
{
    CHECK_NULL_ARGUMENT(batch_normalization_2d, "batch_normalization_2d");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    if (batch_normalization_2d->track_running_stats)
    {
        error = tensor_batch_normalization_2d(x, batch_normalization_2d->weights, batch_normalization_2d->bias, batch_normalization_2d->running_mean, 
                                              batch_normalization_2d->running_variance, y, batch_normalization_2d->inference, batch_normalization_2d->momentum, 
                                              batch_normalization_2d->epsilon);
    }
    else
    {
        error = tensor_batch_normalization_2d(x, batch_normalization_2d->weights, batch_normalization_2d->bias, NULL, NULL, y,
                                              batch_normalization_2d->inference, batch_normalization_2d->momentum, batch_normalization_2d->epsilon);
    }

    if (error)
    {
        return ERROR(ERROR_BATCH_NORMALIZATION, string_create("failed to apply batch normalization 2d."), error);
    }

    return error;
}

nw_error_t *layer_normalization_forward(layer_normalization_t *layer_normalization, tensor_t *x, tensor_t **y)
{
    CHECK_NULL_ARGUMENT(layer_normalization, "layer_normalization");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_layer_normalization(x, layer_normalization->weights, layer_normalization->bias, y, layer_normalization->normalized_shape,
                                       layer_normalization->length, layer_normalization->epsilon);

    if (error)
    {
        return ERROR(ERROR_BATCH_NORMALIZATION, string_create("failed to apply layer normalization."), error);
    }

    return error;
}

nw_error_t *reshape_forward(reshape_t *reshape, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(reshape, "reshape");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_reshape(x, y, reshape->shape, reshape->length);
    if (error)
    {
        return ERROR(ERROR_RESHAPE, string_create("failed to reshape tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *embedding_forward(embedding_t *embedding, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(embedding, "embedding");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    error = tensor_embedding(x, embedding->weights, embedding->vocabulary_counter, y);
    if (error)
    {
        return ERROR(ERROR_EMBEDDING, string_create("failed to embed tensor."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *transformer_embedding_forward(transformer_embedding_t *transformer_embedding, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(transformer_embedding, "transformer_embedding");
    CHECK_NULL_ARGUMENT(x, "x");

    nw_error_t *error = NULL;
    tensor_t *token_embedding = NULL;
    tensor_t *position_embedding = NULL;
    tensor_t *positions = NULL;
    tensor_t *positions_expand = NULL;
    datatype_t datatype = x->buffer->storage->datatype;
    runtime_t runtime = x->buffer->storage->runtime;
    int64_t block_size = x->buffer->view->shape[1];
    void *start = NULL;
    void *stop = NULL;
    void *step = NULL;
    size_t size = datatype_size(datatype);

    start = (void *) malloc(size);
    if (!start)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes", size), NULL);
        goto cleanup;
    }

    stop = (void *) malloc(size);
    if (!stop)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes", size), NULL);
        goto cleanup;
    }

    step = (void *) malloc(size);
    if (!step)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes", size), NULL);
        goto cleanup;
    }

    switch (datatype)
    {
    case FLOAT32:
        *(float32_t *) start = (float32_t) 0.0;
        *(float32_t *) stop = (float32_t) block_size;
        *(float32_t *) step = 1.0;
        break;
    case FLOAT64:
        *(float64_t *) start = (float64_t) 0.0;
        *(float64_t *) stop = (float64_t) block_size;
        *(float64_t *) step = 1.0;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype."), NULL);
        goto cleanup;
    }

    error = tensor_arange(&positions, start, stop, step, runtime, datatype, false, false);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_expand(positions, (int64_t[]){1, block_size}, 2, &positions_expand);
    if (error)
    {
        error = ERROR(ERROR_EXPAND, string_create("failed to expand tensor."), error);
        goto cleanup;
    }

    error = embedding_forward(transformer_embedding->token_embedding, x, &token_embedding);
    if (error)
    {
        error = ERROR(ERROR_EMBEDDING, string_create("failed to embed tensor."), error);
        goto cleanup;
    }

    error = embedding_forward(transformer_embedding->position_embedding, positions_expand, &position_embedding);
    if (error)
    {
        error = ERROR(ERROR_EMBEDDING, string_create("failed to embed tensor."), error);
        goto cleanup;
    }

    error = tensor_addition(token_embedding, position_embedding, y);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

cleanup:

    free(start);
    free(stop);
    free(step);
    tensor_destroy(positions);
    tensor_destroy(positions_expand);

    if (!(x->requires_gradient || transformer_embedding->token_embedding->weights->requires_gradient || transformer_embedding->position_embedding->weights->requires_gradient) || no_gradient)
    {
        tensor_destroy(token_embedding);
        tensor_destroy(position_embedding);
    }

    return error;
}

nw_error_t *causal_multihead_self_attention_forward(causal_multihead_self_attention_t *causal_multihead_self_attention, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(causal_multihead_self_attention, "causal_multihead_self_attention");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    error = tensor_causal_multihead_self_attention(x, causal_multihead_self_attention->input_weights, causal_multihead_self_attention->input_bias,
                                                   causal_multihead_self_attention->output_weights, causal_multihead_self_attention->output_bias,
                                                   causal_multihead_self_attention->number_of_heads, causal_multihead_self_attention->dropout_probability,
                                                   causal_multihead_self_attention->inference, y);
    if (error)
    {
        return ERROR(ERROR_ATTENTION, string_create("failed to apply causal multihead self attention."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *residual_block_forward(block_t *block, tensor_t *x, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(block, "block");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *z = NULL;

    error = block_forward(block, x, &z);
    if (error)
    {
        return ERROR(ERROR_FORWARD, string_create("failed to apply forward operation."), error);
    }

    error = tensor_addition(x, z, y);
    if (error)
    {
        return ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
    }

    if (!(x->requires_gradient || z->requires_gradient) || no_gradient)
    {
        tensor_destroy(z);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

//insert forward
nw_error_t *simple_rnn_cell_forward(void *cell, recurrent_type_t type, tensor_t *x, tensor_t *hidden, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;


    CHECK_NULL_ARGUMENT(cell, "cell");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    switch(type)
    {
        case RNN:
            error = rnn_cell_forward((rnn_cell_t *) cell, x, hidden, y);
            if (error)
            {
                return ERROR(ERROR_RNN, string_create("failed to perform forward pass on the rnn cell."), error);
            }
            break;
        case GRU:
            error = gru_cell_forward((gru_cell_t *)cell, x, hidden, y);
            if (error)
            {
                return ERROR(ERROR_RNN, string_create("failed to perform forward pass on the gru cell."), error);
            }
            break;
        case LSTM:
            error = lstm_cell_forward((lstm_cell_t *)cell, x, hidden, y);
            if (error)
            {
                return ERROR(ERROR_RNN, string_create("failed to perform forward pass on the lstm cell."), error);
            }
            break;
        default:
            return ERROR(ERROR_RNN, string_create("Unknown recurrent cell type."), error);
    }

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;
}

nw_error_t *rnn_cell_forward(rnn_cell_t *rnn, tensor_t *x, tensor_t *hidden, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;
    CHECK_NULL_ARGUMENT(rnn, "rnn");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *input_w = NULL;
    tensor_t *input_w_b = NULL;
    tensor_t *hidden_w = NULL;
    tensor_t *hidden_w_b = NULL;

    error = tensor_matrix_multiplication(x, rnn->weight_ih, &input_w);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(input_w, rnn->bias_ih, &input_w_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_matrix_multiplication(hidden, rnn->weight_hh, &hidden_w);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(hidden_w, rnn->bias_hh, &hidden_w_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    switch(rnn->activation)
    {
        case ACTIVATION_TANH:
            error = tensor_tanh(hidden_w_b, y);
            if (error)
            {
                error = ERROR(ERROR_TANH, string_create("failed to perform tanh operation."), error);
                goto cleanup;
            }
            break;
        case ACTIVATION_LEAKY_RECTIFIED_LINEAR:
            error = tensor_rectified_linear(hidden_w_b, y);
            if (error)
            {
                error = ERROR(ERROR_RECTIFIED_LINEAR, string_create("failed to perform relu operation."), error);
                goto cleanup;
            }
            break;
        default:
            error = ERROR(ERROR_RNN, string_create("unknown activation function for rnn."), error);
            goto cleanup;
    }
   
    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;
    
    return error;

cleanup:
    tensor_destroy(input_w);
    tensor_destroy(input_w_b);
    tensor_destroy(hidden_w);
    tensor_destroy(hidden_w_b);
    return error;
}

nw_error_t *gru_cell_forward(gru_cell_t *gru, tensor_t *x, tensor_t *hidden, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(gru, "gru");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;
    tensor_t *input_r_w = NULL;
    tensor_t *input_r_w_b = NULL;
    tensor_t *hidden_r_w = NULL;
    tensor_t *hidden_r_w_b = NULL;
    tensor_t *pre_sigmoid_r = NULL;
    tensor_t *r_gate = NULL;

    tensor_t *input_z_w = NULL;
    tensor_t *input_z_w_b = NULL;
    tensor_t *hidden_z_w = NULL;
    tensor_t *hidden_z_w_b = NULL;
    tensor_t *pre_sigmoid_z = NULL;
    tensor_t *z_gate = NULL;

    tensor_t *input_n_w = NULL;
    tensor_t *input_n_w_b = NULL;
    tensor_t *hidden_n_w = NULL;
    tensor_t *hidden_n_w_b = NULL;
    tensor_t *hidden_n_reset = NULL;
    tensor_t *pre_tanh_n = NULL;
    tensor_t *n_gate = NULL;

    tensor_t *one_constant = NULL;
    tensor_t *one_minus_z = NULL;
    tensor_t *final_hidden_1 = NULL;
    tensor_t *final_hidden_2 = NULL;
    void *scalar_1 = NULL;
    datatype_t datatype = x->buffer->storage->datatype;
    runtime_t runtime = x->buffer->storage->runtime;
    size_t size = datatype_size(datatype);

    // reset gate
    error = tensor_matrix_multiplication(gru->weight_ir, x, &input_r_w);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(input_r_w, gru->bias_ir, &input_r_w_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_matrix_multiplication(gru->weight_hr, hidden, &hidden_r_w);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(hidden_r_w, gru->bias_ir, &hidden_r_w_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(input_r_w_b, hidden_r_w_b, &pre_sigmoid_r);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_sigmoid(pre_sigmoid_r, &r_gate);
    if (error)
    {
        error = ERROR(ERROR_SIGMOID, string_create("failed to perform sigmoid operation."), error);
        goto cleanup;
    }

    // update gate
    error = tensor_matrix_multiplication(gru->weight_iz, x, &input_z_w);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(input_z_w, gru->bias_iz, &input_z_w_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_matrix_multiplication(gru->weight_hz, hidden, &hidden_z_w);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(hidden_z_w, gru->bias_iz, &hidden_z_w_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(input_z_w_b, hidden_z_w_b, &pre_sigmoid_z);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }
    error = tensor_sigmoid(pre_sigmoid_z, &z_gate);
    if (error)
    {
        error = ERROR(ERROR_SIGMOID, string_create("failed to perform sigmoid operation."), error);
        goto cleanup;
    }

    // new gate
    error = tensor_matrix_multiplication(gru->weight_in, x, &input_n_w);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }
    error = tensor_addition(input_n_w, gru->bias_in, &input_n_w_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_matrix_multiplication(gru->weight_hn, hidden, &hidden_n_w);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(hidden_n_w, gru->bias_in, &hidden_n_w_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(r_gate, hidden_n_w_b, &hidden_n_reset);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }
    error = tensor_addition(input_n_w_b, hidden_n_reset, &pre_tanh_n);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_tanh(pre_tanh_n, &n_gate);
    if (error)
    {
        error = ERROR(ERROR_TANH, string_create("failed to perform tanh operation."), error);
        goto cleanup;
    }

    scalar_1 = malloc(size);
    if (!scalar_1)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    switch(datatype)
    {
    case FLOAT32:
        *(float32_t *) scalar_1 = (float32_t) 1;
        break;
    case FLOAT64:
        *(float64_t *) scalar_1 = (float64_t) 1;
        break;
    default:
        error = ERROR(ERROR_DATATYPE, string_create("unknown datatype %d.", (int)datatype), NULL);
        break;
    }

    error = tensor_constant(scalar_1, datatype, runtime, true, true, &one_constant);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create tensor."), error);
        goto cleanup;
    }

    error = tensor_subtraction(one_constant, z_gate, &one_minus_z);
    if (error)
    {
        error = ERROR(ERROR_SUBTRACTION, string_create("failed to subtract tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(one_minus_z, n_gate, &final_hidden_1);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(z_gate, hidden, &final_hidden_2);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(final_hidden_1, final_hidden_2, y);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }
    
    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;

    return error;

cleanup:
    free(scalar_1);
    tensor_destroy(input_r_w);
    tensor_destroy(input_r_w_b);
    tensor_destroy(hidden_r_w);
    tensor_destroy(hidden_r_w_b);
    tensor_destroy(pre_sigmoid_r);
    tensor_destroy(r_gate);

    tensor_destroy(input_z_w);
    tensor_destroy(input_z_w_b);
    tensor_destroy(hidden_z_w);
    tensor_destroy(hidden_z_w_b);
    tensor_destroy(pre_sigmoid_z);
    tensor_destroy(z_gate);

    tensor_destroy(input_n_w);
    tensor_destroy(input_n_w_b);
    tensor_destroy(hidden_n_w);
    tensor_destroy(hidden_n_w_b);
    tensor_destroy(hidden_n_reset);
    tensor_destroy(pre_tanh_n);
    tensor_destroy(n_gate);

    tensor_destroy(one_constant);
    tensor_destroy(one_minus_z);
    tensor_destroy(final_hidden_1);
    tensor_destroy(final_hidden_2); 
    return error;
}

nw_error_t *lstm_cell_forward(lstm_cell_t *lstm, tensor_t *x, tensor_t *hidden, tensor_t **y)
{
    CHECK_NULL_ARGUMENT(lstm, "lstm");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");

    nw_error_t *error = NULL;

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;
    
    tensor_t *i_gate_w_x = NULL;
    tensor_t *i_gate_w_x_b = NULL;
    tensor_t *i_gate_w_h = NULL;
    tensor_t *i_gate_w_h_b = NULL;
    tensor_t *i_gate_pre_sigmoid = NULL;
    tensor_t *i_gate = NULL;

    tensor_t *f_gate_w_x = NULL;
    tensor_t *f_gate_w_x_b = NULL;
    tensor_t *f_gate_w_h = NULL;
    tensor_t *f_gate_w_h_b = NULL;
    tensor_t *f_gate_pre_sigmoid = NULL;
    tensor_t *f_gate = NULL;

    tensor_t *o_gate_w_x = NULL;
    tensor_t *o_gate_w_x_b = NULL;
    tensor_t *o_gate_w_h = NULL;
    tensor_t *o_gate_w_h_b = NULL;
    tensor_t *o_gate_pre_sigmoid = NULL;
    tensor_t *o_gate = NULL;

    tensor_t *g_gate_w_x = NULL;
    tensor_t *g_gate_w_x_b = NULL;
    tensor_t *g_gate_w_h = NULL;
    tensor_t *g_gate_w_h_b = NULL;
    tensor_t *g_gate_pre_tanh = NULL;
    tensor_t *g_gate = NULL;

    tensor_t *f_c_mult = NULL;
    tensor_t* i_g_mult = NULL;
    tensor_t *c_tanh = NULL;

    // input gate
    error = tensor_matrix_multiplication(lstm->weight_xi, x, &i_gate_w_x);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(i_gate_w_x, lstm->bias_xi, &i_gate_w_x_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_matrix_multiplication(lstm->weight_hi, hidden, &i_gate_w_h);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(i_gate_w_h, lstm->bias_hi, &i_gate_w_h_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(i_gate_w_x_b, i_gate_w_h_b, &i_gate_pre_sigmoid);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_sigmoid(i_gate_pre_sigmoid, &i_gate);
    if (error)
    {
        error = ERROR(ERROR_SIGMOID, string_create("failed to perform sigmoid operation."), error);
        goto cleanup;
    }

    // forget gate
    error = tensor_matrix_multiplication(lstm->weight_xf, x, &f_gate_w_x);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(f_gate_w_x, lstm->bias_xf, &f_gate_w_x_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_matrix_multiplication(lstm->weight_hf, hidden, &f_gate_w_h);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(f_gate_w_h, lstm->bias_hf, &f_gate_w_h_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(f_gate_w_x_b, f_gate_w_h_b, &f_gate_pre_sigmoid);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_sigmoid(f_gate_pre_sigmoid, &f_gate);
    if (error)
    {
        error = ERROR(ERROR_SIGMOID, string_create("failed to perform sigmoid operation."), error);
        goto cleanup;
    }

    // output gate
    error = tensor_matrix_multiplication(lstm->weight_xo, x, &o_gate_w_x);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(o_gate_w_x, lstm->bias_xo, &o_gate_w_x_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_matrix_multiplication(lstm->weight_ho, hidden, &o_gate_w_h);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(o_gate_w_h, lstm->bias_ho, &o_gate_w_h_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(o_gate_w_x_b, o_gate_w_h_b, &o_gate_pre_sigmoid);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_sigmoid(o_gate_pre_sigmoid, &o_gate);
    if (error)
    {
        error = ERROR(ERROR_SIGMOID, string_create("failed to perform sigmoid operation."), error);
        goto cleanup;
    }

    // candidate cell state
    error = tensor_matrix_multiplication(lstm->weight_xc, x, &g_gate_w_x);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(g_gate_w_x, lstm->bias_xc, &g_gate_w_x_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_matrix_multiplication(lstm->weight_hc, hidden, &g_gate_w_h);
    if (error)
    {
        error = ERROR(ERROR_MATRIX_MULTIPLICATION, string_create("failed to matrix multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(g_gate_w_h, lstm->bias_hc, &g_gate_w_h_b);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_addition(g_gate_w_x_b, g_gate_w_h_b, &g_gate_pre_tanh);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    error = tensor_sigmoid(g_gate_pre_tanh, &g_gate);
    if (error)
    {
        error = ERROR(ERROR_SIGMOID, string_create("failed to perform sigmoid operation."), error);
        goto cleanup;
    }

    // cell state
    error = tensor_multiplication(f_gate, lstm->cell_state, &f_c_mult);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    error = tensor_multiplication(i_gate, g_gate, &i_g_mult);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    tensor_destroy(lstm->cell_state);
    error = tensor_addition(f_c_mult, i_g_mult, &lstm->cell_state);
    if (error)
    {
        error = ERROR(ERROR_ADDITION, string_create("failed to add tensors."), error);
        goto cleanup;
    }

    // hidden state
    error = tensor_tanh(lstm->cell_state, &c_tanh);
    if (error)
    {
        error = ERROR(ERROR_TANH, string_create("failed to perform tanh operation."), error);
        goto cleanup;
    }

    error = tensor_multiplication(o_gate, c_tanh, y);
    if (error)
    {
        error = ERROR(ERROR_MULTIPLICATION, string_create("failed to multiply tensors."), error);
        goto cleanup;
    }

    // TODO update cell_state

    return error;

cleanup:
    tensor_destroy(i_gate_w_x);
    tensor_destroy(i_gate_w_x_b);
    tensor_destroy(i_gate_w_h);
    tensor_destroy(i_gate_w_h_b);
    tensor_destroy(i_gate_pre_sigmoid);
    tensor_destroy(i_gate);

    tensor_destroy(f_gate_w_x);
    tensor_destroy(f_gate_w_x_b);
    tensor_destroy(f_gate_w_h);
    tensor_destroy(f_gate_w_h_b);
    tensor_destroy(f_gate_pre_sigmoid);
    tensor_destroy(f_gate);

    tensor_destroy(o_gate_w_x);
    tensor_destroy(o_gate_w_x_b);
    tensor_destroy(o_gate_w_h);
    tensor_destroy(o_gate_w_h_b);
    tensor_destroy(o_gate_pre_sigmoid);
    tensor_destroy(o_gate);

    tensor_destroy(g_gate_w_x);
    tensor_destroy(g_gate_w_x_b);
    tensor_destroy(g_gate_w_h);
    tensor_destroy(g_gate_w_h_b);
    tensor_destroy(g_gate_pre_tanh);
    tensor_destroy(g_gate);

    tensor_destroy(f_c_mult);
    tensor_destroy(i_g_mult);
    tensor_destroy(c_tanh);
    return error;
}

nw_error_t *simple_rnn_layer_forward(rnn_layer_t *layer, tensor_t *x, tensor_t *hidden, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("hidden", hidden);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(hidden, "hidden");

    nw_error_t *error = NULL;
    int64_t batch_size = x->buffer->view->shape[0];
    int64_t features_size = x->buffer->view->shape[2];
    int64_t rank = x->buffer->view->rank;
    bool_t bidirectional = layer->bidirectional;
    tensor_t *prev_output = hidden;
    tensor_t *output = NULL;
    string_t index = NULL; 
    void *cell = NULL;
    tensor_t *cell_output = NULL;
    tensor_t *input = NULL;
    tensor_t *tmp_output = NULL;
    tensor_t *final_hidden = NULL;

    for (int64_t i = 0; i < layer->hidden_size; ++i)
    {
        index = string_create("%lu", i);
        error = map_get(layer->cells, index, &cell);
        if (error)
        {
            error = ERROR(ERROR_GET, string_create("failed to get tensor."), error);
            goto cleanup;
        }

        error = tensor_slice(x, &input, (int64_t[]){0, batch_size, i, i+1, 0, features_size}, 2*rank);
        if (error)
        {
            error = ERROR(ERROR_SLICE, string_create("failed to slice tensor."), error);
            goto cleanup;
        }

        error = simple_rnn_cell_forward(cell, layer->type, input, prev_output, &cell_output);
        if (error)
        {
            error = ERROR(ERROR_RNN, string_create("failed to perform forward operation on the rnn cell"), error);
            goto cleanup;
        }

        if (bidirectional)
        {
            if (i == layer->hidden_size - 1)
            {
                    final_hidden = cell_output;
            } else {
                    tensor_destroy(cell_output); 
            }  
        } else {
            if (i == 0)
            {
                output = cell_output;
                cell_output = NULL;
            } else {
                error = tensor_concatenation(output, cell_output, &tmp_output, (int64_t) 1);
                if (error)
                {
                    error = ERROR(ERROR_CONCATENATION, string_create("failed to concatenate tensors."), error);
                    goto cleanup;
                }

                tensor_destroy(output);
                output = tmp_output;
                tmp_output = NULL;
            }
            tensor_destroy(cell_output);
            cell_output = NULL; 
        }
        string_destroy(index);
        tensor_destroy(input);
        input = NULL;
    } 

   if (bidirectional)
   {
        prev_output = final_hidden;

        for (int64_t j = layer->hidden_size - 1; j > -1; --j)
        {
            index = string_create("%lu", j);
            error = map_get(layer->cells, index, &cell);
            if (error)
            {
                error = ERROR(ERROR_GET, string_create("failed to get tensor."), error);
                goto cleanup;
            }
        
            error = tensor_slice(x, &input, (int64_t[]){0, batch_size, j, j+1, 0, features_size}, 2*rank);
            if (error)
            {
                error = ERROR(ERROR_SLICE, string_create("failed to slice tensor."), error);
                goto cleanup;
            }

            error = simple_rnn_cell_forward(cell, layer->type, input, prev_output, &cell_output);
            if (error)
            {
                error = ERROR(ERROR_RNN, string_create("failed to perform forward operation on the rnn cell"), error);
                goto cleanup;
            }
            
            if (j == 0)
            {
                output = cell_output;
                cell_output = NULL;
            } else {
                error = tensor_concatenation(output, cell_output, &tmp_output, (int64_t) 1);
                if (error)
                {
                    error = ERROR(ERROR_CONCATENATION, string_create("failed to concatenate tensors."), error);
                    goto cleanup;
                }

                tensor_destroy(output);
                output = tmp_output;
                tmp_output = NULL;
            }

            tensor_destroy(cell_output);
            cell_output = NULL; 
            string_destroy(index);
        }
        string_destroy(index);
        tensor_destroy(input);
        input = NULL;
   } 

   *y = output;

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;
    return error;

cleanup:
    tensor_destroy(output);
    tensor_destroy(cell_output);
    tensor_destroy(input);
    tensor_destroy(final_hidden);
    tensor_destroy(tmp_output);
    return error;
}

nw_error_t *simple_rnn_stack_forward(rnn_stack_t *stack, tensor_t *x, tensor_t *hidden, bool_t inference, tensor_t **y)
{
    PRINTLN_DEBUG_LOCATION("input");
    PRINTLN_DEBUG_TENSOR("x", x);
    PRINTLN_DEBUG_TENSOR("hidden", hidden);
    PRINT_DEBUG_NEWLINE;

    CHECK_NULL_ARGUMENT(stack, "stack");
    CHECK_NULL_ARGUMENT(x, "x");
    CHECK_NULL_ARGUMENT(y, "y");
    CHECK_NULL_ARGUMENT(hidden, "hidden");

    nw_error_t *error = NULL;
    int64_t batch_size = x->buffer->view->shape[0];

    int64_t hidden_size = stack->hidden_size;
    int64_t num_layers = stack->num_layers;
    int64_t rank = hidden->buffer->view->rank;
    int64_t num_directions = stack->bidirectional?2:1;
    string_t index;
    tensor_t *layer_outputs = NULL;
    rnn_layer_t *layer = NULL;
    tensor_t *input = x;
    tensor_t *input_hidden = hidden;
    tensor_t *tmp_dropout_output = NULL;

    if (hidden->buffer->view->rank != 3 ||
          hidden->buffer->view->shape[0] != num_layers * num_directions ||
          hidden->buffer->view->shape[1] != batch_size ||
          hidden->buffer->view->shape[2] != hidden_size)
    {
        return ERROR(ERROR_SHAPE, string_create("Invalid hidden state shape, expected: (num_layers * num_directions, batch_size, hidden_size)"), NULL);
    }

    for (int64_t i = 0; i < num_layers; ++i)
    {
        index = string_create("%lu", i);
        error = map_get(stack->layers, index, (void *) &layer);
        if (error)
        {
            error = ERROR(ERROR_GET, string_create("failed to get tensor."), error);
            goto cleanup;
        }

        error = tensor_slice(input, &input_hidden, (int64_t[]){i, i+1, 0, batch_size, 0, hidden_size}, 2*rank); 
        if (error)
        {
            error = ERROR(ERROR_SLICE, string_create("failed to slice tensor."), error);
            goto cleanup;
        }

       error = simple_rnn_layer_forward(layer, input, input_hidden, &layer_outputs);
        if (error)
        {
            error = ERROR(ERROR_RNN, string_create("failed to perform forward operation on the rnn layer"), error);
            goto cleanup;
        }

       if (i != num_layers - 1)      
       {
            error = tensor_dropout(layer_outputs, &tmp_dropout_output, stack->dropout, inference);
            if (error)
            {
                error = ERROR(ERROR_DROPOUT, string_create("failed to apply dropout."), error);
                goto cleanup;
            }

            layer_outputs = tmp_dropout_output;
            tmp_dropout_output = NULL;
       } 

       string_destroy(index);
       input = layer_outputs;
    }

    *y = layer_outputs;

    PRINTLN_DEBUG_LOCATION("output");
    PRINTLN_DEBUG_TENSOR("y", *y);
    PRINT_DEBUG_NEWLINE;
    return error;

cleanup:
    tensor_destroy(input_hidden);
    tensor_destroy(layer_outputs);
    tensor_destroy(tmp_dropout_output);
    return error;
}

nw_error_t *model_inference(model_t *model, bool_t inference)
{
    CHECK_NULL_ARGUMENT(model, "model");

    nw_error_t *error = block_inference(model->block, inference);
    if (error)
    {
        return ERROR(ERROR_SET, string_create("failed to set inference flag."), error);
    }

    return error;
}

nw_error_t *block_inference(block_t *block, bool_t inference)
{
    CHECK_NULL_ARGUMENT(block, "block");

    nw_error_t *error = NULL;

    for (int64_t i = 0; i < block->depth; ++i)
    {
        switch (block->layers[i]->transform_type)
        {
        case DROPOUT:
            block->layers[i]->transform->dropout->inference = inference;
            break;
        case BATCH_NORMALIZATION_2D:
            block->layers[i]->transform->batch_normalization_2d->inference = inference;
            break;
        case CAUSAL_MULTIHEAD_SELF_ATTENTION:
            block->layers[i]->transform->causal_multihead_self_attention->inference = inference;
            break;
        case BLOCK:
            error = block_inference(block->layers[i]->transform->block, inference);
            if (error)
            {
                return ERROR(ERROR_SET, string_create("failed to set inference flag."), error);
            }
            break;
        default:
            break;
        }
    }

    return error;
}

nw_error_t *model_save(model_t *model, string_t path)
{
    CHECK_NULL_ARGUMENT(model, "model");
    CHECK_NULL_ARGUMENT(path, "path");

    FILE *file = fopen(path, "wb");
    if (!file)
    {
        return ERROR(ERROR_FILE, string_create("failed to open file."), NULL);
    }

    nw_error_t *error = block_save(model->block, file);
    if (error)
    {
        fclose(file);
        return ERROR(ERROR_SAVE, string_create("failed to save block."), error);
    }    

    fclose(file);

    return error;
}

nw_error_t *block_save(block_t *block, FILE *file)
{
    CHECK_NULL_ARGUMENT(block, "block");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&block->depth, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    for (int64_t i = 0; i < block->depth; ++i)
    {
        error = layer_save(block->layers[i], file);
        if (error)
        {
            return ERROR(ERROR_SAVE, string_create("failed to save layer."), error);
        }
    }

    return error;
}

nw_error_t *layer_save(layer_t *layer, FILE *file)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&layer->transform_type, sizeof(transform_type_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    switch (layer->transform_type)
    {
    case LINEAR:
        error = linear_save(layer->transform->linear, file);
        break;
    case CONVOLUTION_2D:
    case CONVOLUTION_TRANSPOSE_2D:
        error = convolution_2d_save(layer->transform->convolution_2d, file);
        break;
    case MAX_POOLING_2D:
    case AVERAGE_POOLING_2D:
        error = pooling_2d_save(layer->transform->pooling_2d, file);
        break;
    case DROPOUT:
        error = dropout_save(layer->transform->dropout, file);
        break;
    case BATCH_NORMALIZATION_2D:
        error = batch_normalization_2d_save(layer->transform->batch_normalization_2d, file);
        break;
    case RESHAPE:
        error = reshape_save(layer->transform->reshape, file);
        break;
    case LAYER_NORMALIZATION:
        error = layer_normalization_save(layer->transform->layer_normalization, file);
        break;
    case EMBEDDING:
        error = embedding_save(layer->transform->embedding, file);
        break;
    case TRANSFORMER_EMBEDDING:
        error = transformer_embedding_save(layer->transform->transformer_embedding, file);
        break;
    case CAUSAL_MULTIHEAD_SELF_ATTENTION:
        error = causal_multihead_self_attention_save(layer->transform->causal_multihead_self_attention, file);
        break;
    case ACTIVATION:
        error = activation_save(layer->transform->activation, file);
        break;
    case RESIDUAL_BLOCK:
    case BLOCK:
        error = block_save(layer->transform->block, file);
        break;
    default:
        error = ERROR(ERROR_LAYER_TYPE, string_create("unknown transform type %d.", (int) layer->transform_type), NULL);
        break;
    }

    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save transform."), error);
    }
    
    return error;
}

nw_error_t *linear_save(linear_t *linear, FILE *file)
{
    CHECK_NULL_ARGUMENT(linear, "linear");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    error = tensor_save(linear->weights, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(linear->bias, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    return error;
}

nw_error_t *convolution_2d_save(convolution_2d_t *convolution_2d, FILE *file)
{
    CHECK_NULL_ARGUMENT(convolution_2d, "convolution_2d");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    error = tensor_save(convolution_2d->kernel, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(convolution_2d->bias, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    if (!fwrite(&convolution_2d->padding, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&convolution_2d->stride, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    return error;
}

nw_error_t *pooling_2d_save(pooling_2d_t *pooling_2d, FILE *file)
{
    CHECK_NULL_ARGUMENT(pooling_2d, "pooling_2d");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&pooling_2d->kernel, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&pooling_2d->padding, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&pooling_2d->stride, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    return error;
}

nw_error_t *dropout_save(dropout_t *dropout, FILE *file)
{
    CHECK_NULL_ARGUMENT(dropout, "dropout");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&dropout->datatype, sizeof(datatype_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(dropout->probability, datatype_size(dropout->datatype), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&dropout->inference, sizeof(bool_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    return error;
}

nw_error_t *batch_normalization_2d_save(batch_normalization_2d_t *batch_normalization_2d, FILE *file)
{
    CHECK_NULL_ARGUMENT(batch_normalization_2d, "batch_normalization_2d");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&batch_normalization_2d->datatype, sizeof(datatype_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&batch_normalization_2d->track_running_stats, sizeof(bool_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&batch_normalization_2d->inference, sizeof(bool_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(batch_normalization_2d->momentum, datatype_size(batch_normalization_2d->datatype), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(batch_normalization_2d->epsilon, datatype_size(batch_normalization_2d->datatype), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    error = tensor_save(batch_normalization_2d->weights, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(batch_normalization_2d->bias, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(batch_normalization_2d->running_mean, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(batch_normalization_2d->running_variance, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }
    
    return error;
}

nw_error_t *layer_normalization_save(layer_normalization_t *layer_normalization, FILE *file)
{
    CHECK_NULL_ARGUMENT(layer_normalization, "layer_normalization");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&layer_normalization->datatype, sizeof(datatype_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(layer_normalization->epsilon, datatype_size(layer_normalization->datatype), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&layer_normalization->length, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(layer_normalization->normalized_shape, sizeof(int64_t), layer_normalization->length, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    error = tensor_save(layer_normalization->weights, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(layer_normalization->bias, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    return error;
}

nw_error_t *reshape_save(reshape_t *reshape, FILE *file)
{
    CHECK_NULL_ARGUMENT(reshape, "reshape");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&reshape->length, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(reshape->shape, sizeof(int64_t), reshape->length, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    return error;
}

nw_error_t *embedding_save(embedding_t *embedding, FILE *file)
{
    CHECK_NULL_ARGUMENT(embedding, "embedding");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&embedding->vocabulary_size, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&embedding->embedding_size, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    error = tensor_save(embedding->vocabulary_counter, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(embedding->weights, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    return error;
}

nw_error_t *transformer_embedding_save(transformer_embedding_t *transformer_embedding, FILE *file)
{
    CHECK_NULL_ARGUMENT(transformer_embedding, "transformer_embedding");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    error = embedding_save(transformer_embedding->token_embedding, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save embedding."), error);
    }

    error = embedding_save(transformer_embedding->position_embedding, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save embedding."), error);
    }

    return error;
}

nw_error_t *causal_multihead_self_attention_save(causal_multihead_self_attention_t *causal_multihead_self_attention, FILE *file)
{
    CHECK_NULL_ARGUMENT(causal_multihead_self_attention, "causal_multihead_self_attention");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    if (!fwrite(&causal_multihead_self_attention->datatype, sizeof(datatype_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(causal_multihead_self_attention->dropout_probability, datatype_size(causal_multihead_self_attention->datatype), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&causal_multihead_self_attention->embedding_size, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&causal_multihead_self_attention->number_of_heads, sizeof(int64_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    if (!fwrite(&causal_multihead_self_attention->inference, sizeof(bool_t), 1, file))
    {
        return ERROR(ERROR_WRITE, string_create("failed to write to file."), NULL);
    }

    error = tensor_save(causal_multihead_self_attention->input_weights, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(causal_multihead_self_attention->input_bias, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(causal_multihead_self_attention->output_weights, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    error = tensor_save(causal_multihead_self_attention->output_bias, file);
    if (error)
    {
        return ERROR(ERROR_SAVE, string_create("failed to save tensor."), error);
    }

    return error;
}

//insert save

nw_error_t *model_load(model_t **model, string_t path)
{
    CHECK_NULL_ARGUMENT(model, "model");
    CHECK_NULL_ARGUMENT(path, "path");

    FILE *file = fopen(path, "rb");
    if (!file)
    {
        return ERROR(ERROR_FILE, string_create("failed to open file."), NULL);
    }
    
    *model = (model_t *) malloc(sizeof(model_t));
    if (!*model)
    {
        fclose(file);
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(model_t)), NULL);
    }
    (*model)->block = NULL;


    nw_error_t *error = block_load(&(*model)->block, file);
    if (error)
    {
        fclose(file);
        model_destroy(*model);
        return ERROR(ERROR_LOAD, string_create("failed to load block."), error);
    }    

    fclose(file);

    return error;
}

nw_error_t *block_load(block_t **block, FILE *file)
{
    CHECK_NULL_ARGUMENT(block, "block");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *block = (block_t *) malloc(sizeof(block_t));
    if (!*block)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(block_t)), NULL);
        goto cleanup;
    }
    (*block)->layers = NULL;
    (*block)->depth = 0;

    if (!fread(&(*block)->depth, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read to file."), NULL);
        goto cleanup;
    }

    (*block)->layers = (layer_t **) malloc((*block)->depth * sizeof(layer_t *));
    if (!(*block)->layers)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", (*block)->depth * sizeof(layer_t *)), NULL);
        goto cleanup;
    }

    for (int64_t i = 0; i < (*block)->depth; ++i)
    {
        error = layer_load(&(*block)->layers[i], file);
        if (error)
        {
            error = ERROR(ERROR_LOAD, string_create("failed to load layer."), error);
            goto cleanup;
        }
    }

    return error;

cleanup:

    block_destroy(*block);

    return error;
}

nw_error_t *layer_load(layer_t **layer, FILE *file)
{
    CHECK_NULL_ARGUMENT(layer, "layer");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *layer = (layer_t *) malloc(sizeof(layer_t));
    if (!*layer)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(layer_t)), NULL);
        goto cleanup;
    }
    (*layer)->transform = NULL;

    if (!fread(&(*layer)->transform_type, sizeof(transform_type_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    (*layer)->transform = (transform_t *) malloc(sizeof(transform_t));
    if (!(*layer)->transform)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(transform_t)), NULL);
        goto cleanup;
    }

    switch ((*layer)->transform_type)
    {
    case LINEAR:
        error = linear_load(&(*layer)->transform->linear, file);
        break;
    case CONVOLUTION_2D:
    case CONVOLUTION_TRANSPOSE_2D:
        error = convolution_2d_load(&(*layer)->transform->convolution_2d, file);
        break;
    case MAX_POOLING_2D:
    case AVERAGE_POOLING_2D:
        error = pooling_2d_load(&(*layer)->transform->pooling_2d, file);
        break;
    case DROPOUT:
        error = dropout_load(&(*layer)->transform->dropout, file);
        break;
    case BATCH_NORMALIZATION_2D:
        error = batch_normalization_2d_load(&(*layer)->transform->batch_normalization_2d, file);
        break;
    case RESHAPE:
        error = reshape_load(&(*layer)->transform->reshape, file);
        break;
    case LAYER_NORMALIZATION:
        error = layer_normalization_load(&(*layer)->transform->layer_normalization, file);
        break;
    case EMBEDDING:
        error = embedding_load(&(*layer)->transform->embedding, file);
        break;
    case TRANSFORMER_EMBEDDING:
        error = transformer_embedding_load(&(*layer)->transform->transformer_embedding, file);
        break;
    case CAUSAL_MULTIHEAD_SELF_ATTENTION:
        error = causal_multihead_self_attention_load(&(*layer)->transform->causal_multihead_self_attention, file);
        break;
    case ACTIVATION:
        error = activation_load(&(*layer)->transform->activation, file);
        break;
    case RESIDUAL_BLOCK:
    case BLOCK:
        error = block_load(&(*layer)->transform->block, file);
        break;
    default:
        error = ERROR(ERROR_LAYER_TYPE, string_create("unknown transform type %d.", (int) (*layer)->transform_type), NULL);
        break;
    }

    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed load transform."), error);
        goto cleanup;
    }

    return error;

cleanup:

    layer_destroy(*layer);

    return error;
}

nw_error_t *linear_load(linear_t **linear, FILE *file)
{
    CHECK_NULL_ARGUMENT(linear, "linear");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;
    
    *linear = (linear_t *) malloc(sizeof(linear_t));
    if (!*linear)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(linear_t)), NULL);
        goto cleanup;
    }

    (*linear)->weights = NULL;
    (*linear)->bias = NULL;
    
    error = tensor_load(&(*linear)->weights, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*linear)->bias, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    return error;

cleanup:

    linear_destroy(*linear);

    return error;
}

nw_error_t *convolution_2d_load(convolution_2d_t **convolution_2d, FILE *file)
{
    CHECK_NULL_ARGUMENT(convolution_2d, "convolution_2d");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *convolution_2d = (convolution_2d_t *) malloc(sizeof(convolution_2d_t));
    if (!*convolution_2d)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(convolution_2d_t)), NULL);
        goto cleanup;
    }

    (*convolution_2d)->kernel = NULL;
    (*convolution_2d)->bias = NULL;

    error = tensor_load(&(*convolution_2d)->kernel, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*convolution_2d)->bias, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    if (!fread(&(*convolution_2d)->padding, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*convolution_2d)->stride, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    return error;

cleanup:

    convolution_2d_destroy(*convolution_2d);

    return error;
}

nw_error_t *pooling_2d_load(pooling_2d_t **pooling_2d, FILE *file)
{
    CHECK_NULL_ARGUMENT(pooling_2d, "pooling_2d");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *pooling_2d = (pooling_2d_t *) malloc(sizeof(pooling_2d_t));
    if (!*pooling_2d)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(pooling_2d_t)), NULL);
        goto cleanup;
    }

    if (!fread(&(*pooling_2d)->kernel, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*pooling_2d)->padding, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*pooling_2d)->stride, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    return error;

cleanup:

    pooling_2d_destroy(*pooling_2d);

    return error;
}

nw_error_t *dropout_load(dropout_t **dropout, FILE *file)
{
    CHECK_NULL_ARGUMENT(dropout, "dropout");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *dropout = (dropout_t *) malloc(sizeof(dropout_t));
    if (!*dropout)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(dropout_t)), NULL);
        goto cleanup;
    }
    (*dropout)->probability = NULL;

    if (!fread(&(*dropout)->datatype, sizeof(datatype_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    (*dropout)->probability = (void *) malloc(datatype_size((*dropout)->datatype));
    if (!(*dropout)->probability)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size((*dropout)->datatype)), NULL);
        goto cleanup;
    }

    if (!fread((*dropout)->probability, datatype_size((*dropout)->datatype), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*dropout)->inference, sizeof(bool_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    return error;

cleanup:

    dropout_destroy(*dropout);

    return error;
}

nw_error_t *batch_normalization_2d_load(batch_normalization_2d_t **batch_normalization_2d, FILE *file)
{
    CHECK_NULL_ARGUMENT(batch_normalization_2d, "batch_normalization_2d");
    CHECK_NULL_ARGUMENT(file, "file");

    *batch_normalization_2d = (batch_normalization_2d_t *) malloc(sizeof(batch_normalization_2d_t));
    if (!*batch_normalization_2d)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(batch_normalization_2d_t)), NULL);
    }

    nw_error_t *error = NULL;

    (*batch_normalization_2d)->momentum = NULL;
    (*batch_normalization_2d)->epsilon = NULL;
    (*batch_normalization_2d)->bias = NULL;
    (*batch_normalization_2d)->weights = NULL;
    (*batch_normalization_2d)->running_mean = NULL;
    (*batch_normalization_2d)->running_variance = NULL;

    if (!fread(&(*batch_normalization_2d)->datatype, sizeof(datatype_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*batch_normalization_2d)->track_running_stats, sizeof(bool_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*batch_normalization_2d)->inference, sizeof(bool_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    size_t size = datatype_size((*batch_normalization_2d)->datatype);

    (*batch_normalization_2d)->momentum = (void *) malloc(size);
    if (!(*batch_normalization_2d)->momentum)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    (*batch_normalization_2d)->epsilon = (void *) malloc(size);
    if (!(*batch_normalization_2d)->epsilon)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    if (!fread((*batch_normalization_2d)->momentum, size, 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread((*batch_normalization_2d)->epsilon, size, 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    error = tensor_load(&(*batch_normalization_2d)->weights, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*batch_normalization_2d)->bias, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*batch_normalization_2d)->running_mean, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*batch_normalization_2d)->running_variance, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }
    
    return error;

cleanup:

    batch_normalization_2d_destroy(*batch_normalization_2d);

    return error;
}

nw_error_t *layer_normalization_load(layer_normalization_t **layer_normalization, FILE *file)
{
    CHECK_NULL_ARGUMENT(layer_normalization, "layer_normalization");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *layer_normalization = (layer_normalization_t *) malloc(sizeof(layer_normalization_t));
    if (!*layer_normalization)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(layer_normalization_t)), NULL);
        goto cleanup;
    }

    (*layer_normalization)->epsilon = NULL;
    (*layer_normalization)->bias = NULL;
    (*layer_normalization)->weights = NULL;
    (*layer_normalization)->normalized_shape = NULL;

    if (!fread(&(*layer_normalization)->datatype, sizeof(datatype_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    size_t size = datatype_size((*layer_normalization)->datatype);
    
    (*layer_normalization)->epsilon = (void *) malloc(size);
    if (!(*layer_normalization)->epsilon)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    if (!fread((*layer_normalization)->epsilon, size, 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*layer_normalization)->length, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    size = (*layer_normalization)->length * sizeof(int64_t);

    (*layer_normalization)->normalized_shape = (int64_t *) malloc(size);
    if (!(*layer_normalization)->normalized_shape)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
        goto cleanup;
    }

    if (!fread((*layer_normalization)->normalized_shape, sizeof(int64_t), (*layer_normalization)->length, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    error = tensor_load(&(*layer_normalization)->weights, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*layer_normalization)->bias, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    return error;

cleanup:

    layer_normalization_destroy(*layer_normalization);

    return error;
}

nw_error_t *reshape_load(reshape_t **reshape, FILE *file)
{
    CHECK_NULL_ARGUMENT(reshape, "reshape");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *reshape = (reshape_t *) malloc(sizeof(reshape_t));
    if (!*reshape)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(reshape_t)), NULL);
        goto cleanup;
    }

    (*reshape)->shape = NULL;

    if (!fread(&(*reshape)->length, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    (*reshape)->shape = (int64_t *) malloc((*reshape)->length * sizeof(int64_t));
    if (!(*reshape)->shape)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", (*reshape)->length * sizeof(int64_t)), NULL);
        goto cleanup;
    }

    if (!fread((*reshape)->shape, sizeof(int64_t), (*reshape)->length, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    return error;

cleanup:

    reshape_destroy(*reshape);

    return error;
}

nw_error_t *embedding_load(embedding_t **embedding, FILE *file)
{
    CHECK_NULL_ARGUMENT(embedding, "embedding");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *embedding = (embedding_t *) malloc(sizeof(embedding_t));
    if (!*embedding)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(embedding_t)), NULL);
        goto cleanup;
    }

    (*embedding)->vocabulary_counter = NULL;
    (*embedding)->weights = NULL;

    if (!fread(&(*embedding)->vocabulary_size, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*embedding)->embedding_size, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    error = tensor_load(&(*embedding)->vocabulary_counter, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*embedding)->weights, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    return error;

cleanup:

    embedding_destroy(*embedding);

    return error;
}

nw_error_t *transformer_embedding_load(transformer_embedding_t **transformer_embedding, FILE *file)
{
    CHECK_NULL_ARGUMENT(transformer_embedding, "transformer_embedding");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *transformer_embedding = (transformer_embedding_t *) malloc(sizeof(transformer_embedding_t));
    if (!*transformer_embedding)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(transformer_embedding_t)), NULL);
    }

    (*transformer_embedding)->token_embedding = NULL;
    (*transformer_embedding)->position_embedding = NULL;

    error = embedding_load(&(*transformer_embedding)->token_embedding, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load embedding."), error);
        goto cleanup;
    }

    error = embedding_load(&(*transformer_embedding)->position_embedding, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load embedding."), error);
        goto cleanup;
    }

    return error;

cleanup:

    transformer_embedding_destroy(*transformer_embedding);

    return error;
}

nw_error_t *causal_multihead_self_attention_load(causal_multihead_self_attention_t **causal_multihead_self_attention, FILE *file)
{
    CHECK_NULL_ARGUMENT(causal_multihead_self_attention, "causal_multihead_self_attention");
    CHECK_NULL_ARGUMENT(file, "file");

    nw_error_t *error = NULL;

    *causal_multihead_self_attention = (causal_multihead_self_attention_t *) malloc(sizeof(causal_multihead_self_attention_t));
    if (!*causal_multihead_self_attention)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", sizeof(causal_multihead_self_attention_t)), NULL);
        goto cleanup;
    }

    (*causal_multihead_self_attention)->input_weights = NULL;
    (*causal_multihead_self_attention)->input_bias = NULL;
    (*causal_multihead_self_attention)->output_weights = NULL;
    (*causal_multihead_self_attention)->output_bias = NULL;
    (*causal_multihead_self_attention)->dropout_probability = NULL;

    if (!fread(&(*causal_multihead_self_attention)->datatype, sizeof(datatype_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    (*causal_multihead_self_attention)->dropout_probability = (void *) malloc(datatype_size((*causal_multihead_self_attention)->datatype));
    if (!(*causal_multihead_self_attention)->dropout_probability)
    {
        error = ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", datatype_size((*causal_multihead_self_attention)->datatype)), NULL);
        goto cleanup;
    }

    if (!fread((*causal_multihead_self_attention)->dropout_probability, datatype_size((*causal_multihead_self_attention)->datatype), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*causal_multihead_self_attention)->embedding_size, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*causal_multihead_self_attention)->number_of_heads, sizeof(int64_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    if (!fread(&(*causal_multihead_self_attention)->inference, sizeof(bool_t), 1, file))
    {
        error = ERROR(ERROR_READ, string_create("failed to read from file."), NULL);
        goto cleanup;
    }

    error = tensor_load(&(*causal_multihead_self_attention)->input_weights, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*causal_multihead_self_attention)->input_bias, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*causal_multihead_self_attention)->output_weights, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    error = tensor_load(&(*causal_multihead_self_attention)->output_bias, file);
    if (error)
    {
        error = ERROR(ERROR_LOAD, string_create("failed to load tensor."), error);
        goto cleanup;
    }

    return error;

cleanup:

    causal_multihead_self_attention_destroy(*causal_multihead_self_attention);

    return error;
}

//insert load