#include <iostream>
#include <tuple>
extern "C"
{
#include <check.h>
#include <buffer.h>
#include <tensor.h>
#include <view.h>
#include <errors.h>
#include <datatype.h>
#include <optimizer.h>
#include <layer.h>
#include <function.h>
#include <test_helper.h>
#include <cost.h>
}
#include <test_helper_torch.h>

// class RNN_Torch : public torch::nn::Module {
// public:
//     RNN_Torch(int input_size, int hidden_size, int num_layers) {
//         rnn_layer = torch::nn::RNN(torch::nn::RNNOptions(input_size, hidden_size).num_layers(num_layers).nonlinearity(torch::kReLU));
//     }

//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input, torch::Tensor hx) {
//         // Pass input through the RNN layer
//         torch::Tensor output;
//         std::tie(output, hidden) = rnn_layer->forward(input, hx);

//         return std::make_tuple(output, hidden);
//     }

// private:
//     torch::nn::RNN rnn_layer{nullptr};
//     torch::Tensor hidden;
// };


void test_rnn()
{   
    int input_size = 10; 
    int hidden_size = 3; 
    int num_layers = 1;
    int batch_size = 2;
    int sequence_length = 4;

    // torch rnn
    torch::nn::RNN rnn_torch = torch::nn::RNN(torch::nn::RNNOptions(input_size, hidden_size).num_layers(num_layers).nonlinearity(torch::kReLU).batch_first(true));

    auto input_torch = torch::randn({batch_size, sequence_length, input_size});
    auto hidden_torch = torch::randn({num_layers * 1, batch_size, hidden_size});

    torch::Tensor output, new_hidden;
    std::tie(output, new_hidden) = rnn_torch->forward(input_torch, hidden_torch);

    // NW rnn
    nw_error_t *error = NULL;
    rnn_stack_t *rnn_nw = NULL;    
    parameter_init_t *weight_init, *bias_init;
    tensor_t *output_nw = NULL;
    void *min = NULL, *max = NULL, *dropout_probability = NULL;

    *(float32_t *) min = (float32_t) -math.sqrt(1/hidden_size);
    *(float32_t *) max = (float32_t) math.sqrt(1/hidden_size);
    *(float32_t *) dropout_probability = (float32_t) 0.0;

    error = uniform_parameter_init(&weight_init, min, max, FLOAT32);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create parameter initializer."), error);
        goto cleanup;
    }

    error = uniform_parameter_init(&bias_init, min, max, FLOAT32);
    if (error)
    {
        error = ERROR(ERROR_CREATE, string_create("failed to create parameter initializer."), error);
        goto cleanup;
    }

    tensor_t *input_nw = torch_to_tensor(input_torch, MKL_RUNTIME, FLOAT32);
    tensor_t *hidden_nw = torch_to_tensor(hidden_torch, MKL_RUNTIME, FLOAT32);
    tensor_t *output_torch_nw = torch_to_tensor(output_torch, MKL_RUNTIME, FLOAT32);
    tensor_t *output_hidden_torch_nw = torch_to_tensor(new_hidden, MKL_RUNTIME, FLOAT32);

    error = rnn_stack_create(&rnn_nw, RNN, num_layers, batch_size, input_size, hidden_size, MKL_RUNTIME, FLOAT32, false, weight_init, bias_init, hidden_init, ACTIVATION_LEAKY_RECTIFIED_LINEAR);

    error = simple_rnn_stack_forward(rnn_nw, input_nw, hidden_nw, dropout_probability, false, &output_nw);

//     ck_assert_tensor_equiv(output_nw, output_torch_nw);
// cleanup:
//     return error;
}

void setup_rnn(void)
{

}

void teardown_rnn(void)
{

}

START_TEST(test_plain_rnn)
{
    test_rnn();
}
END_TEST

Suite *make_rnn_suite(void)
{
    Suite *s;
    TCase *tc_rnn;

    s = suite_create("Test RNN Suite");

    tc_rnn = tcase_create("Test Plain RNN Case");
    tcase_add_checked_fixture(tc_rnn, setup_rnn, teardown_rnn);
    tcase_add_test(tc_rnn, test_plain_rnn);

    suite_add_tcase(s, tc_rnn);
    return s;
}
int main(void)
{
    // Set seed
    torch::manual_seed(SEED);

    int number_failed;
    SRunner *sr;

    sr = srunner_create(make_rnn_suite());
    srunner_set_fork_status(sr, CK_NOFORK);
    srunner_run_all(sr, CK_VERBOSE);

    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);
    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}