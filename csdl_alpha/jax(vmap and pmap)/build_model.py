from csdl_alpha import Recorder
def build_model(model_cls):
    rec = Recorder(inline=True)
    rec.start()
    model = model_cls()
    rec.stop()
    return rec, model
