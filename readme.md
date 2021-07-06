Readme:

Which Attention Do You Need: Dynamic Routing, Self-attention or Graph Pooling?


#### change to the working directory
```sh
cd ./AttenCapsuleGraphPooling
```

#### choose a mode to run the model
1. Option 1: run capsule dummy mode

```sh
python main_diffpool.py --pooling_method capsule --capsule_mode dummy
``` 

2. Option 2: run capsule capsule mode

```sh
python main_diffpool.py --pooling_method capsule --capsule_mode capsule
```

3. Option 3: run diffpool mode

```sh
python main_diffpool.py --pooling_method diffpool
``` 