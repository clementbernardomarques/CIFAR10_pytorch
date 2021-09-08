# CIFAR10_pytorch

## Train model

Pour entraîner le modèle il convient de 
- Ouvrir un terminal
- Se rendre dans le dossier CIFAR10_pytorch
- créer un environnement virtuel et l'activer
- taper dans le terminal ```pip install -r requirements.txt```
- taper dans le terminal ```python main.py```

Pour modifier les valeurs des hyperparamètres il faut aller dans le fichier main.py et modifier la list "lr_to_test" et "batch_size_to_test"

## Export model as ONNX model

Pour exporter un modèle en ONNX il suffit de lancer le fichier ```export_model.py``` (pour changer les weights du modèle à exporter, il convient de modifier la première ligne du code avec la localisation des dits weights).


## Inference

Pour tester le modèle ONNX sur une image il suffit de taper dans un terminal:
```python inference.py --PATH=<path_of_your_image>```
