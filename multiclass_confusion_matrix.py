import random
def generate_predictions(n, n_classes):
    return [random.randint(1, n_classes) for _ in range(n)]

model_predictions = []
m_models = 4
n_classes = 4

for _ in range(m_models):
    model_predictions.append(generate_predictions(n=100, n_classes=n_classes))

ground_truth = generate_predictions(n=100, n_classes=n_classes)

def calculate_matrix(y_pred, y_true, n_classes):
    matrix = [[0] * n_classes for _ in range(n_classes)]

    # x:= predicted class
    # y:= actual class

    for y_hat, y in zip(y_pred, y_true):
        matrix[y_hat-1][y-1] += 1

    return matrix


print(calculate_matrix(model_predictions[0], ground_truth, n_classes=n_classes))