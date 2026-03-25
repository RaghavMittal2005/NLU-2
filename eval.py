N_GENERATE   = 200
TRAINING_SET = set(n.lower() for n in training_names)

def evaluate_model(model, model_name, n=N_GENERATE, temperature=0.8):
    generated = generate_names_batch(model, n=n, temperature=temperature)
    valid     = [nm for nm in generated if len(nm) >= 3]
    if not valid:
        return {'model': model_name, 'total': 0, 'novel': 0,
                'unique': 0, 'novelty': 0.0, 'diversity': 0.0,
                'generated': []}

    novel     = [nm for nm in valid if nm.lower() not in TRAINING_SET]
    unique    = set(valid)
    novelty   = len(novel)  / len(valid) * 100
    diversity = len(unique) / len(valid) * 100
    return {'model': model_name, 'total': len(valid),
            'novel': len(novel), 'unique': len(unique),
            'novelty': novelty, 'diversity': diversity,
            'generated': valid}

print('Evaluating models (generating 200 names each)...')
results = {}
MODEL_LABELS = [('VanillaRNN', rnn_model),
                ('LSTM',       blstm_model),
                ('RNN+CausalAttn', attn_model)]
for nm, mdl in MODEL_LABELS:
    results[nm] = evaluate_model(mdl, nm)
    print(f'  {nm} → sample: {results[nm]["generated"][:5]}')

print('\n' + '=' * 60)
print('  QUANTITATIVE EVALUATION RESULTS')
print('=' * 60)
print(f'  {"Model":<20} {"Total":>6} {"Novel":>6} {"Unique":>6} '
      f'{"Novelty%":>10} {"Diversity%":>12}')
print('  ' + '-' * 60)
for nm, r in results.items():
    print(f'  {nm:<20} {r["total"]:>6} {r["novel"]:>6} {r["unique"]:>6} '
          f'{r["novelty"]:>9.1f}% {r["diversity"]:>11.1f}%')
print('=' * 60)