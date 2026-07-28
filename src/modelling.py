"""Train and evaluate MOE models."""


def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model
