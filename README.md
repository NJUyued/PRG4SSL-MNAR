# PRG4SSL-MNAR

## Results (e.g. seed=1)

| Dateset | Labels | N0 |gamma|Acc|Setting|Method|
| :-----:| :----: | :----: |:----: |:----: |:----: |:----: |
|CIFAR-10 | 40 | - |-|94.05 |Conventional settings|PRG|
| | 250 | - |- |94.36 |||
| | 4000 | - |- |95.48 |||
| | 40 | - |-|93.79 |Conventional settings|PRG^Last|
| | 250 | - |- |94.76 |||
| | 4000 | - |- |95.75 |||
| | - | - |20 |94.04 |CADR's protocol|PRG|
| | - | - |50 |93.78 |||
| | - | - |100 |94.51 |||
| | - | - |20 |94.74 |CADR's protocol|PRG^Last|
| | - | - |50 |94.74 |||
| | - | - |100 |94.75 |||
| | 40 | 10 |- |93.61 |Ours protocol|PRG|
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
|CIFAR-100  | 400 | - |- |48.67 |Conventional settings|PRG|
|  | 2500 | - |- |69.81|||
|  | 10000 | - |- |76.91 |||
|  | 400 | - |- |48.66 |Conventional settings|PRG^Last|
|  | 2500 | - |- |70.03|||
|  | 10000 | - |- |76.93 |||
|  | - | - |50 |58.57 |CADR's protocol|PRG|
|  | - | - |100 |62.28 |||
|  | - | - |200 |59.33 |||
|  | - | - |50 |60.32 |CADR's protocol|PRG^Last|
|  | - | - |100 |62.13 |||
|  | - | - |200 |58.70 |||
|  | 2500 | 100 |- |57.56 |Ours protocol|PRG|
|  | 2500 | 200 |- |51.21 |||
|mini-ImageNet | 1000| -|- |45.74 |Conventional settings|PRG|
| | 1000| -|- |48.63 |Conventional settings|PRG^Last|
| | -| -|50 |43.72 |CADR's protocol|PRG|
| | -| - |100 |43.74 |||
| | -| -|50 |39.16 |CADR's protocol|PRG^Last|
| | -| - |100 |42.08 |||
| | 1000| 40 |- |40.75 |Ours protocol|||
| | 1000| 80 |- |35.86|||
