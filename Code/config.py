dataset = {"type_choices": ["seismic", "dem", "rock"],
           "degradation_choices": ["Noise0", "Noise1", "Noise3", "Noise5", "Noise7", "Noise9", "Noise1+Blur", "Blind"]}

train_parameters = {
    "seismic": {
        "epochs": 50,
        "batch_sizes": 4,
        "scale": 2,
        "input_size": (128, 128, 1),
        "initial_learning_rate": 0.0005
    },
    "dem": {
        "epochs": 50,
        "batch_sizes": 32,
        "scale": 2,
        "input_size": (32, 32, 1),
        "initial_learning_rate": 0.0005
    },
    "rock": {
        "epochs": 50,
        "batch_sizes": 8,
        "scale": 2,
        "input_size": (128, 128, 1),
        "initial_learning_rate": 0.0005
    },
}

model_parameters = {
    "number_filters": 64,
    "number_groups": 4,
    "number_blockss": 8,}
