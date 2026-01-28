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
- cosa si misura in conformal, a che punto sono arrivati.
- che relazione c'e' tra p(y|c) e p(c|x)? Le varie calibrazioni, calibrazione multilabel, naive bayes, modelli markoviani
- Test Optuna with dry-run