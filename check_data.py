import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
manual = json.loads((ROOT / 'data' / 'university_manual.json').read_text(encoding='utf-8'))
train = json.loads((ROOT / 'data' / 'domain_train.json').read_text(encoding='utf-8'))
test = json.loads((ROOT / 'data' / 'domain_test.json').read_text(encoding='utf-8'))
print(f'Manual sections: {len(manual)}')
print(f'Domain train examples: {len(train)}')
print(f'Domain test examples: {len(test)}')
print('Data check passed.')
