from train import train_model

def test_model_accuracy():
    accuracy = train_model()
    assert accuracy >= 0.80

def test_accuracy_range():
    accuracy = train_model()
    assert 0 <= accuracy <= 1
