![Jogo da Velha](https://www.iberdrola.com/wcorp/gc/prod/pt_BR/comunicacion/machine_learning_mult_1_res/machine_learning_746x419.jpg)
# Jogo da Velha - Aprendizagem de Máquina
## Objetivo

### Desenvolver uma aplicação que possa predizer a porcentagem de vencer de cada jogador à cada turno baseado no classificador de [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) usando o [scikit-learn](https://scikit-learn.org/) em Python 3.6+

    O classificador em questão não se mostrou tão preciso quanto outros classificadores.

* Neural Network Precisão: 99.31%
* Logistic Regression Precisão: 98.26%
* Support Vector Machine Precisão: 97.57%
* Decision Tree Precisão: 94.44%
* K-Nearest Neighbors Precisão: 89.58%
* Naive Bayers Precisão: 63.19%

## Breve Descrição
Ao ser iniciado o aplicativo irá verificar se não existe um modelo que já foi previamente treinado `./data/trained_model.pkl`, caso não encontre ele irá começar filtrar os dados do `./data/tic-tac-toe.csv` que foi devidamente filtrado após ser encontrado no Centro de aprendizagem de máquina e sistemas inteligentes da Califórnia [aqui](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame)
## Documentação :page_facing_up:

### **Classificador**

  * Função responsável por tratar os dados para nós usarmos posteriormente de forma mais eficiente.

    ```py
    def preprocess_inputs(df):
      df = df.copy()
      # codificando os valores negativo / positivo como {0, 1}
      df['V10'] = df['V10'].replace({'negative': 0, 'positive': 1})
      
      # codificando todas as outras colunas como Vn, tal que n = i;
      df = onehot_encode(
          df,
          columns=['V' + str(i) for i in range(1, 10)]
      )
      
      # Split df em x -> y
      y = df['V10'].copy()
      X = df.drop('V10', axis=1).copy()
      
      # Treinando para retornar os valores
      X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)
      
      return X_train, X_test, y_train, y_test
    ```
  * Função auxiliar para a função anterior

    ```py
      def onehot_encode(df, columns):
        df = df.copy()
        for column in columns:
            dummies = pd.get_dummies(df[column], prefix=column)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(column, axis=1)
        return df
    ```
### **Jogo**
![Aplicação](https://i.imgur.com/CazEfWA.gif)
## Inicialização :octocat:

[0] - **Clonando repositório**
```bash
git clone https://github.com/Remato/machine-learning-tic-tac-toe && cd machine-learning-tic-tac-toe
```

[1] - **Instalando requirements.txt**

  ```bash
  pip install requirements.txt
  ```
