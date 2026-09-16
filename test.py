from medmnist import INFO 

info = INFO['organcmnist']
class_names = [info['label'][str(i)] for i in range(len(info['label']))]

print(class_names)