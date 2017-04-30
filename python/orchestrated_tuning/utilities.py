def get_middle_index(values):
    return (len(values) - 1) / 2


def get_middle_value(values):
    return values[get_middle_index(values)]


def update_parameters(method, params):
    for param_name, param_idx in params.iteritems():
        method["cur_params"][param_name] = method["param_values"][param_name][param_idx]
        method["cur_param_idx"][param_name] = param_idx


def pack_method_properties(method, param_values, cur_params, cur_param_idx, cur_fit, cur_coef, callable):
    return {
        "method": method,
        "param_values": param_values,
        "cur_params": cur_params,
        "cur_param_idx": cur_param_idx,
        "cur_fit": cur_fit,
        "cur_coef": cur_coef,
        "callable": callable
    }


def get_cache_key(method_name, params):
    return method_name + "(" + ', '.join(str(param) for param in params) + ")"

def compare_params(model1, model2):
    return model1["cur_param_idx"].values() == model2["cur_param_idx"].values()