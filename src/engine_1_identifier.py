from infralearning.AI_Modelo import AI_Modelo
from infralearning.Dados import Dados


# dados = Dados('/home/victor/infra-learning/images')

dados = Dados('/media/victor/infra-learning/not_interpolated')
modelo = AI_Modelo().create_model_binary()

modelo.run(dados, epochs=10)
modelo.save_model('models/detection.h5')

modelo.show_precision(dados.test)
modelo.plot_loss_graph()
modelo.plot_accuracy_graph()