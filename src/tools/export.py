import json


def export_power_transformer(transformer, dir):
    transformer_params = {
        'lambdas': list(transformer.lambdas_),
    }
    with open(dir, 'w') as f:
        f.write(json.dumps(transformer_params))


def export_standard_scaler(scaler, dir):
    scaler_params = {
        'means': list(scaler.mean_),
        'scales': list(scaler.scale_),
    }
    with open(dir, 'w') as f:
        f.write(json.dumps(scaler_params))