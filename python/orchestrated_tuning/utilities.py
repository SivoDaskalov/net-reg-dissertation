import pandas as pd
custom_start_points = None


def get_middle_param(grid):
    idx = (len(grid) - 1) / 2
    val = grid[idx]
    return idx, val


def get_initial_param(grid, setup, method, param_name):
    if custom_start_points is None:
        return get_middle_param(grid)
    if not setup in custom_start_points:
        return get_middle_param(grid)
    if not method in custom_start_points[setup]:
        return get_middle_param(grid)
    if not param_name in custom_start_points[setup][method]:
        return get_middle_param(grid)
    # There is a custom start point for the current context
    target = custom_start_points[setup][method][param_name]
    val = min(grid, key=lambda x: abs(x - target))
    idx = grid.index(val)
    return idx, val


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


def load_custom_start_points(results_dump_url):
    global custom_start_points
    results = pd.read_csv(results_dump_url, index_col=0)
    custom_start_points = {}
    for index, row in results.iterrows():
        setup = row["setup"]
        method = row["model"]
        params = [key_value.strip().split('=') for key_value in row["params"].split(',')]
        if not setup in custom_start_points:
            custom_start_points[setup] = {}
        if not method in custom_start_points[setup]:
            custom_start_points[setup][method] = {}
        custom_start_points[setup][method] = {param[0]: float(param[1]) for param in params}
    print("Custom orchestrated tuning start points loaded")
    return custom_start_points