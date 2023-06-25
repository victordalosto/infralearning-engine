from domain.AI_Modelo import AI_Modelo
from domain.Dados import Dados

dados = Dados('D:\\imgs\\mount\\CLASSIFIER')
modelo = AI_Modelo().create_model_categorial(num_classes=7)

modelo.run(train_data=dados.train, validation_data=dados.validation, test_data=dados.test)
modelo.save_model('models\\classification_2.h5')
modelo.show_precision(dados.test)
