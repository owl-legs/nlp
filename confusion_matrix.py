import random
def generate_predictions(n):
    return [random.randint(0, 1) for _ in range(n)]

model_predictions = []
m_models = 4

for _ in range(m_models):
    model_predictions.append(generate_predictions(n=100))

ground_truth = generate_predictions(n=100)

#precision := tp / (tp + fp)
#negative_predictive_value := tn / (tn + fn)
#sensitivity := tp / (tp + fn)
#specificity := tn / (tn + fp)

def calculate_stats(y_pred, y):
    tp, fp, tn, fn = 0, 0, 0, 0
    for y_hat, y_true in zip(y_pred, y):
        if y_hat == y_true == 1:
            tp += 1
        elif y_hat == y_true == 0:
            tn += 1
        elif y_hat == 1 and y_true == 0:
            fp += 1
        elif y_hat == 0 and y_true == 1:
            fn += 1
    return {'tp':tp, 'fp':fp, 'tn':tn, 'fn':fn}

def calculate_matrix(y_pred, y):
    stats = calculate_stats(y_pred, y)
    matrix = [[stats['tp'], stats['fn']], [stats['fp'], stats['tn']]]
    return matrix, {'precision': stats['tp']/(stats['tp'] + stats['fp']),
                    'negative_pred_value': stats['tn']/(stats['tn'] + stats['fn']),
                    'sensitivity':stats['tp']/(stats['tp'] + stats['fn']),
                    'specificity':stats['tn']/(stats['tn'] + stats['fp'])}

model_stats = {}
for i in range(m_models):
    matrix, stats = calculate_matrix(model_predictions[i], ground_truth)
    model_stats[f'''model_{i}'''] = {'matrix':matrix,
                                     'core_stats':stats}

print(model_stats)


