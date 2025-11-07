# Notebooks



## Create a Jupyter Kernel from the Poetry environment

1. Make sure the poetry environment is activated:

```bash
poetry shell
```

2. Create the kernel:

```bash
python -m ipykernel install --user --name pirnns --display-name "Python (pirnns)"
```

3. Verify the kernel was created:

```bash
jupyter kernelspec list
```