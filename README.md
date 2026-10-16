# YOLO 26

Ce repo rassemble les scripts d'inférence (main.py), d'évaluation (eval.py) et d'entrainement (train.py) pour YOLO 26.
Il présente également quelques datasets d'évaluation utiles et le dossier runs qui contient tous les résultats d'entraînement, les poids des modèles et les paramètres d'entraînements. Parmi tous ces modèles, les plus intéressants sont runs/segment/train-24/weights/best.pt (meilleur modèle pour la segmentation de parcelles) et runs/segment/300000_multi/weights/best.pt (meilleur modèle pour la segmentation de parcelles, bâtiments intérieurs et bâtiments extérieurs).
Dans runs/mlflow/, est disponible la base de données MLFlow (backend sqlite) qui contient les informations de certains de ces entraînements. 