from mindscape.evaluation.schemas import ErrorType, PredictionRecord


def evaluate(example, prediction, backend):
    correct = type(prediction.answer) is type(example.target_answer) and prediction.answer == example.target_answer
    valid, goal, error = backend.assess(example, prediction.answer, prediction.trajectory)
    goal = goal and correct
    grounded = correct and valid is True and goal
    if prediction.error_type:
        error = ErrorType(prediction.error_type).value
    elif not correct:
        error = "arithmetic_error" if prediction.answer is not None else "unsupported_answer"
    elif grounded:
        error = "correct"
    return PredictionRecord(example.example_id, prediction.answer, example.target_answer,
                            correct, prediction.trajectory, valid, goal, grounded, error)
