#include <iostream>
#include <tuple>
extern "C"
{
#include <check.h>
#include <buffer.h>
#include <view.h>
#include <errors.h>
#include <datatype.h>
#include <test_helper.h>
}
#include <torch/torch.h>

#define CASES 4

nw_error_t *error;

buffer_t *buffers_x[RUNTIMES][DATATYPES][CASES];
buffer_t *buffers_y[RUNTIMES][DATATYPES][CASES];
buffer_t *returned_buffers[RUNTIMES][DATATYPES][CASES];
buffer_t *expected_buffers[RUNTIMES][DATATYPES][CASES];

view_t *views_x[RUNTIMES][DATATYPES][CASES];
view_t *views_y[RUNTIMES][DATATYPES][CASES];
view_t *returned_views[RUNTIMES][DATATYPES][CASES];
view_t *expected_views[RUNTIMES][DATATYPES][CASES];

torch::Tensor tensors_x[RUNTIMES][DATATYPES][CASES];
torch::Tensor tensors_y[RUNTIMES][DATATYPES][CASES];

std::vector<int64_t> shapes[CASES] = {
    {1},
    {1},
    {1},
    {1},
};

void setup(void)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        runtime_create_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                buffers_x[i][j][k] = NULL;
                buffers_y[i][j][k] = NULL;
                returned_buffers[i][j][k] = NULL;
                expected_buffers[i][j][k] = NULL;

                views_x[i][j][k] = NULL;
                views_y[i][j][k] = NULL;
                returned_views[i][j][k] = NULL;
                expected_views[i][j][k] = NULL;
            }

            for (int k = 0; k < CASES; ++k)
            {
                switch ((datatype_t) j)
                {
                case FLOAT32:
                    tensors_x[i][j][k] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat32));
                    tensors_y[i][j][k] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat32));
                    break;
                case FLOAT64:
                    tensors_x[i][j][k] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat64));
                    tensors_y[i][j][k] = torch::randn(shapes[k], torch::TensorOptions().dtype(torch::kFloat64));
                    break;
                default:
                    ck_abort_msg("unknown datatype.");
                }

                error = view_create(&views_x[i][j][k], 
                                    (uint64_t) tensors_x[i][j][k].storage_offset(),
                                    (uint64_t) tensors_x[i][j][k].ndimension(),
                                    (uint64_t *) tensors_x[i][j][k].sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&buffers_x[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      views_x[i][j][k],
                                      (void *) tensors_x[i][j][k].data_ptr(),
                                      (uint64_t) tensors_x[i][j][k].numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = view_create(&views_y[i][j][k], 
                                    (uint64_t) tensors_y[i][j][k].storage_offset(),
                                    (uint64_t) tensors_y[i][j][k].ndimension(),
                                    (uint64_t *) tensors_y[i][j][k].sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&buffers_y[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      views_y[i][j][k],
                                      (void *) tensors_y[i][j][k].data_ptr(),
                                      (uint64_t) tensors_y[i][j][k].numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = view_create(&returned_views[i][j][k],
                                    (uint64_t) tensors_x[i][j][k].storage_offset(),
                                    (uint64_t) tensors_x[i][j][k].ndimension(),
                                    (uint64_t *) tensors_x[i][j][k].sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&returned_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      returned_views[i][j][k],
                                      NULL,
                                      (uint64_t) tensors_x[i][j][k].numel(),
                                      true);
                ck_assert_ptr_null(error);
            }
        }
    }
}

void teardown(void)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        runtime_destroy_context((runtime_t) i);
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                buffer_destroy(buffers_x[i][j][k]);
                buffer_destroy(buffers_y[i][j][k]);
                buffer_destroy(returned_buffers[i][j][k]);
                buffer_destroy(expected_buffers[i][j][k]);
            }
        }
    }
    error_print(error);
    error_destroy(error);
}

START_TEST(test_addition)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::add(tensors_x[i][j][k], tensors_y[i][j][k]);

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_addition(buffers_x[i][j][k], buffers_y[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

START_TEST(test_subtraction)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::subtract(tensors_x[i][j][k], tensors_y[i][j][k]);

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_subtraction(buffers_x[i][j][k], buffers_y[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

START_TEST(test_multiplication)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::mul(tensors_x[i][j][k], tensors_y[i][j][k]);

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_multiplication(buffers_x[i][j][k], buffers_y[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

START_TEST(test_division)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::div(tensors_x[i][j][k], tensors_y[i][j][k]);

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_division(buffers_x[i][j][k], buffers_y[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

START_TEST(test_power)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::pow(tensors_x[i][j][k], tensors_y[i][j][k]);

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_power(buffers_x[i][j][k], buffers_y[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

START_TEST(test_compare_equal)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::eq(tensors_x[i][j][k], tensors_y[i][j][k]).to(tensors_x[i][j][k].dtype());

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_compare_equal(buffers_x[i][j][k], buffers_y[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

START_TEST(test_compare_greater)
{
    for (int i = 0; i < RUNTIMES; ++i)
    {
        for (int j = 0; j < DATATYPES; ++j)
        {
            for (int k = 0; k < CASES; ++k)
            {
                torch::Tensor expected_tensor = torch::gt(tensors_x[i][j][k], tensors_y[i][j][k]).to(tensors_x[i][j][k].dtype());

                error = view_create(&expected_views[i][j][k],
                                    (uint64_t) expected_tensor.storage_offset(),
                                    (uint64_t) expected_tensor.ndimension(),
                                    (uint64_t *) expected_tensor.sizes().data(),
                                    NULL);
                ck_assert_ptr_null(error);
                error = buffer_create(&expected_buffers[i][j][k],
                                      (runtime_t) i,
                                      (datatype_t) j,
                                      expected_views[i][j][k],
                                      (void *) expected_tensor.data_ptr(),
                                      (uint64_t) expected_tensor.numel(),
                                      true);
                ck_assert_ptr_null(error);

                error = runtime_compare_greater(buffers_x[i][j][k], buffers_y[i][j][k], returned_buffers[i][j][k]);
                ck_assert_ptr_null(error);

                ck_assert_buffer_eq(returned_buffers[i][j][k], expected_buffers[i][j][k]);
            }
        }
    }
}
END_TEST

Suite *make_buffer_binary_suite(void)
{
    Suite *s;
    TCase *tc_binary;

    s = suite_create("Test Buffer Binary Suite");

    tc_binary = tcase_create("Buffer Binary Case");
    tcase_add_checked_fixture(tc_binary, setup, teardown);
    tcase_add_test(tc_binary, test_addition);
    tcase_add_test(tc_binary, test_subtraction);
    tcase_add_test(tc_binary, test_multiplication);
    tcase_add_test(tc_binary, test_division);
    tcase_add_test(tc_binary, test_power);
    tcase_add_test(tc_binary, test_compare_equal);
    tcase_add_test(tc_binary, test_compare_greater);

    suite_add_tcase(s, tc_binary);

    return s;
}

int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_buffer_binary_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}