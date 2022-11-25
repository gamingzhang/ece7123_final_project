import numpy as np
import copy
import torch


def prune_model(model_original, solution, filter_nums):
    model = copy.deepcopy(model_original)
    parameters = model.state_dict()

    parameters_new = copy.deepcopy(parameters)
    cum = 0
    for i, (k, v) in enumerate(model.state_dict().items()):
        param_idx = int(i / 2)  # 0,0,1,1,2,2,3,3...
        cum = sum(filter_nums[:param_idx])  # increase cumulative variable
        # Get the code for current kernels
        code_idx = np.where(solution[cum:cum + filter_nums[param_idx]] == 1)[0]

        if k.find('weight') != -1:  # Conv layer
            if param_idx == 0:  # the first layer, no channel change on its input, only deal with first dim of filters (2nd dim=3 always)
                parameters_new[k] = parameters_new[k][code_idx, ...]
            elif param_idx > 0:  # begin with second layer, deal with change of its input channels (2nd dim) and pruning (1st dim)
                code_idx_last = np.where(solution[cum - filter_nums[param_idx - 1]:cum] == 1)[0]  # code of previous layer
                parameters_new[k] = parameters_new[k][code_idx, ...][:, code_idx_last, ...]  # change 1st and 2nd dim together

        if k.find('bias') != -1:  # bias layer
            parameters_new[k] = parameters_new[k][code_idx, ...]

    # get all conv weights and biases
    conv_weights = []
    conv_biases = []
    for k, v in parameters_new.items():
        if k.find('weight') != -1:
            conv_weights.append(v)
        if k.find('bias') != -1:
            conv_biases.append(v)

    # Find all conv layers in the model and assign with new weights

    ConvLayers = []
    for layer in list(model.children()):
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            ConvLayers.append(layer)

    # Update properties of each conv layer
    for conv_layer, conv_weight, conv_bias in zip(ConvLayers, conv_weights, conv_biases):
        conv_layer.weight.data = conv_weight
        conv_layer.bias.data = conv_bias

    return model
