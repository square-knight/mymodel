

def getModelName(model_identifier=None):
    model_name = 'finger-model'
    if model_identifier is not None:
        model_name = model_name + '-' + model_identifier
    return model_name