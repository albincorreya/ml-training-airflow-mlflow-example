import pytest


@pytest.mark.ml
def test_audio_cnn_model_build_graph_summary_pass(audio_cnn_model):
    print(audio_cnn_model.build_graph().summary())

