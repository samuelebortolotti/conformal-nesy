# Conformal NeSy

## Training

```sh
    python -m conformal CONF train mnistadd --epochs 20 lenet
```

## Testing

```sh
    python -m conformal CONF test mnistadd --epochs 20 lenet
```

> Note: CUB and Animals with Attribute do not have Logic
> 
> Better move to Clevr?

- interazione sul concetto da parte dell'utente, o sulla label o sui concetti.
- cub tassonomia per famiglia diversa, definire un set di concetti categorici 28.
- dataset gerarchici
- se non supervisionati, la logica deve essere applicata con la permutazione