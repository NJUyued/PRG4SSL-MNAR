# LDG4SSL-MNAR

## Results (e.g. seed=1)

| Dateset | Labels | N0 |gamma|Acc|Setting|Method|
| :-----:| :----: | :----: |:----: |:----: |:----: |:----: |
|CIFAR-10 | 40 | - |-|94.05 |conventional settings|PRG|
| | 250 | - |- |94.36 |||
| | 4000 | - |- |95.48 |||
| | 40 | - |-|93.78 |conventional settings|PRG^Last|
| | 250 | - |- |94.69 |||
| | 4000 | - |- |94.67 |||
| | - | - |20 |94.04 |CADR's protocol|PRG|
| | - | - |50 |93.78 |||
| | - | - |100 |94.51 |||
| | - | - |20 |94.74 |CADR's protocol|PRG^Last|
| | - | - |50 |94.74 |||
| | - | - |100 |94.75 |||
| | 40 | 10 |- |93.61 |ours protocol|PRG|
| | 40 | 20 |- |93.39 |||
| | 40 | 10 |2 |90.25 |||
| | 40 | 10 |5 |82.84 |||
| | 100 | 40 |5 |79.58 |||
| | 100 | 40 |10 |78.61 |||
| | 250 | 100 |- |93.76 |||
| | 250 | 200 |- |91.65 |||
|  | DARP | 100 |1 |94.41 |DARP's protocol|PRG|
|  | DARP | 100 |50 |78.28 |||
|  | DARP | 100 |150 |75.21 |||
|  | DARP (reversed) | 100 |100 |80.86 |||
|CIFAR-100  | 400 | - |- |48.67 |conventional settings|PRG|
|  | 2500 | - |- |69.81|||
|  | 10000 | - |- |76.91 |||
|  | - | - |50 |58.57 |CADR's protocol||
|  | - | - |100 |62.28 |||
|  | - | - |200 |59.33 |||
|  | 2500 | 100 |- |57.56 |ours protocol||
|  | 2500 | 200 |- |51.21 |||
|mini-ImageNet | 1000| -|- |45.74 |conventional settings|PRG|
| | 1000| -|- |48.63 |conventional settings|PRG^Last|
| | -| -|50 |43.72 |CADR's protocol|||
| | -| - |100 |43.74 |||
| | 1000| 40 |- |40.75 |ours protocol|||
| | 1000| 80 |- |35.86|||
