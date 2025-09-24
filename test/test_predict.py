import pytest
from src.predict import predict_single

def test_predict_missing_model():
    # Si no hay pipeline, debe lanzar FileNotFoundError
    with pytest.raises(FileNotFoundError):
        predict_single({"pclass": 3, "sex": "male", "age": 22, "sibsp": 1, "parch": 0, "fare": 7.25, "embarked": "S"})
