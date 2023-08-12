#include <mkl_runtime.h>
#include <mkl.h>

error_t *mkl_memory_allocate(void **pp, size_t size)
{
    CHECK_NULL_ARGUMENT(pp, "pp");

    *pp = mkl_malloc(size, ALIGNMENT);
    if (*pp == NULL)
    {
        return ERROR(ERROR_MEMORY_ALLOCATION, string_create("failed to allocate %zu bytes.", size), NULL);
    }

    return NULL;
}

void mkl_memory_free(void *p)
{
    mkl_free(p);
}

void mkl_exponential(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        vsExpI((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        vdExpI((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

void mkl_logarithm(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        vsLnI((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        vdLnI((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

void mkl_sine(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        vsSinI((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        vdSinI((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

void mkl_cosine(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        vsCosI((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        cdCosI((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

void mkl_square_root(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        vsSqrtI((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        vdSqrtI((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

void mkl_reciprocal(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        vsInvI((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        vdInvI((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

void mkl_copy(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cblas_scopy((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        cblas_dcopy((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

void mkl_negation(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cblas_saxpby((int) n, (float32_t) -1.0, &((float32_t *) x_data)[x_offset], (int) x_stride, (float32_t) 0.0, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        cblas_daxpby((int) n, (float64_t) -1.0, &((float64_t *) x_data)[x_offset], (int) x_stride, (float64_t) 0.0, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

void mkl_rectified_linear(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_stride, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        mkl_rectified_linear_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride);
        break;
    case FLOAT64:
        mkl_rectified_linear_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride);
        break;
    default:
        break;
    }
}

static void mkl_rectified_linear_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data, int y_stride)
{
    for (int i = 0; i < n; i++)
    {
        float32_t value = x_data[i * x_stride];
        y_data[i * y_stride] = (value > 0.0) ? value : (float32_t) 0.0; 
    }
}

static void mkl_rectified_linear_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data, int y_stride)
{
    for (int i = 0; i < n; i++)
    {
        float64_t value = x_data[i * x_stride];
        y_data[i * y_stride] = (value > 0.0) ? value : (float64_t) 0.0; 
    }
}

void mkl_addition(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, const void *y_data, uint32_t y_stride, uint32_t y_offset, void *z_data, uint32_t z_stride, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        vsAddI((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        vdAddI((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

void mkl_subtraction(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, const void *y_data, uint32_t y_stride, uint32_t y_offset, void *z_data, uint32_t z_stride, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        vsSubI((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        vdSubI((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

void mkl_multiplication(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, const void *y_data, uint32_t y_stride, uint32_t y_offset, void *z_data, uint32_t z_stride, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        vsMulI((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        vdMulI((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

void mkl_division(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, const void *y_data, uint32_t y_stride, uint32_t y_offset, void *z_data, uint32_t z_stride, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        vsDivI((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        vdDivI((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

void mkl_power(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, const void *y_data, uint32_t y_stride, uint32_t y_offset, void *z_data, uint32_t z_stride, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        vsPowI((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        vdPowI((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

void mkl_compare_equal(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, const void *y_data, uint32_t y_stride, uint32_t y_offset, void *z_data, uint32_t z_stride, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        mkl_compare_equal_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        mkl_compare_equal_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

static void mkl_compare_equal_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    for (int i = 0; i < n; i++)
    {
        z_data[i * z_stride] = (x_data[i * x_stride] == y_data[i * y_stride]) ? (float32_t) 1.0 : (float32_t) 0.0;
    }
}

static void mkl_compare_equal_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    for (int i = 0; i < n; i++)
    {
        z_data[i * z_stride] = (x_data[i * x_stride] == y_data[i * y_stride]) ? (float64_t) 1.0 : (float64_t) 0.0;
    }
}

void mkl_compare_greater(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, const void *y_data, uint32_t y_stride, uint32_t y_offset, void *z_data, uint32_t z_stride, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        mkl_compare_greater_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset], (int) y_stride, &((float32_t *) z_data)[z_offset], (int) z_stride);
        break;
    case FLOAT64:
        mkl_compare_greater_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset], (int) y_stride, &((float64_t *) z_data)[z_offset], (int) z_stride);
        break;
    default:
        break;
    }
}

static void mkl_compare_greater_float32(int n, const float32_t *x_data, int x_stride, const float32_t *y_data, int y_stride, float32_t *z_data, int z_stride)
{
    for (int i = 0; i < n; i++)
    {
        z_data[i * z_stride] = (x_data[i * x_stride] > y_data[i * y_stride]) ? (float32_t) 1.0 : (float32_t) 0.0;
    }
}

static void mkl_compare_greater_float64(int n, const float64_t *x_data, int x_stride, const float64_t *y_data, int y_stride, float64_t *z_data, int z_stride)
{
    for (int i = 0; i < n; i++)
    {
        z_data[i * z_stride] = (x_data[i * x_stride] > y_data[i * y_stride]) ? (float64_t) 1.0 : (float64_t) 0.0;
    }
}

void mkl_matrix_multiplication(datatype_t datatype, uint32_t m, uint32_t k, uint32_t n, bool_t x_transpose, bool_t y_transpose, const void *x_data, uint32_t x_offset, const void *y_data, uint32_t y_offset, void *z_data, uint32_t z_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        cblas_sgemm(CblasRowMajor, (x_transpose) ? CblasNoTrans: CblasTrans, (y_transpose) ? CblasNoTrans : CblasTrans, (int) m, (int) n, (int) k, 1.0,
                    &((float32_t *) x_data)[x_offset], (int) m, &((float32_t *) y_data)[y_offset], (int) k, 0.0, &((float32_t *) z_data)[z_offset], (int) m);
        break;
    case FLOAT64:
        cblas_dgemm(CblasRowMajor, (x_transpose) ? CblasNoTrans: CblasTrans, (y_transpose) ? CblasNoTrans : CblasTrans, (int) m, (int) n, (int) k, 1.0, 
                    &((float64_t *) x_data)[x_offset], (int) m, &((float64_t *) y_data)[y_offset], (int) k, 0.0, &((float64_t *) z_data)[z_offset], (int) m);
        break;
    default:
        break;
    }
}

void mkl_summation(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        mkl_summation_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset]);
        break;
    case FLOAT64:
        mkl_summation_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset]);
        break;
    default:
        break;
    }
}

static void mkl_summation_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data)
{
    float32_t temp = 1.0;
    *y_data = cblas_sdot(n, x_data, x_stride, &temp, (int) 1);
}

static void mkl_summation_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data)
{
    float64_t temp = 1.0;
    *y_data = cblas_ddot(n, x_data, x_stride, &temp, (int) 1);
}

void mkl_maximum(datatype_t datatype, uint32_t n, const void *x_data, uint32_t x_stride, uint32_t x_offset, void *y_data, uint32_t y_offset)
{
    switch (datatype)
    {
    case FLOAT32:
        mkl_maximum_float32((int) n, &((float32_t *) x_data)[x_offset], (int) x_stride, &((float32_t *) y_data)[y_offset]);
        break;
    case FLOAT64:
        mkl_maximum_float64((int) n, &((float64_t *) x_data)[x_offset], (int) x_stride, &((float64_t *) y_data)[y_offset]);
        break;
    default:
        break;
    }
}

static void mkl_maximum_float32(int n, const float32_t *x_data, int x_stride, float32_t *y_data)
{
    float32_t maximum = *x_data;
    for (int i = 1; i < n; i++)
    {
        float32_t candidate = x_data[i];
        if (maximum < candidate)
        {
            maximum = candidate;
        }
    }
    *y_data = maximum;
}

static void mkl_maximum_float64(int n, const float64_t *x_data, int x_stride, float64_t *y_data)
{
    float64_t maximum = *x_data;
    for (int i = 1; i < n; i++)
    {
        float64_t candidate = x_data[i];
        if (maximum < candidate)
        {
            maximum = candidate;
        }
    }
    *y_data = maximum;
}