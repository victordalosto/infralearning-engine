from infralearning.AI_Modelo import AI_Modelo
from infralearning.Dados import Dados

dados = Dados(data_dir='/home/victor/infra-learning/10000_35_mount/CLASSIFICATION')
modelo = AI_Modelo().create_model_categorial(dados)

modelo.run(dados)
modelo.save_model('models/classification.h5')
modelo.show_precision(dados.test)
modelo.plot_loss_graph()
modelo.plot_accuracy_graph()